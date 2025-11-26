import pandas as pd
import numpy as np
import joblib
import pickle
import logging
from typing import Dict, List, Any

class LoanApprovalInferencePipeline:
    def __init__(self, model_path: str, feature_names_path: str = None):
        """Initialize the inference pipeline with trained model"""
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)
        
        if feature_names_path:
            with open(feature_names_path, 'rb') as f:
                self.expected_features = pickle.load(f)
            self.logger.info(f"Loaded {len(self.expected_features)} expected features")
        else:
            self.expected_features = None
            self.logger.warning("No feature names file provided")
        self.payload_cols = ['id', 'address_state', 'application_type', 'emp_length', 'emp_title',
                           'grade', 'home_ownership', 'issue_date', 'last_credit_pull_date',
                           'last_payment_date','next_payment_date', 'member_id',
                           'purpose', 'sub_grade', 'term', 'verification_status', 'annual_income',
                           'dti', 'installment', 'int_rate', 'loan_amount', 'total_acc',
                           'total_payment']
        
        self.drop_cols = ['id', 'member_id', 'emp_title', 'issue_date',
                         'last_payment_date', 'next_payment_date', 'last_credit_pull_date',
                         'total_payment', 'loan_status', 'application_type']
        
        self.emp_length_short_term = ['< 1 year', '1 year', '2 years','3 years', '4 years', '5 years']
        
        self.grade_map = {g: i for i, g in enumerate("ABCDEFG", start=1)}
        
        self.subgrade_map = {}
        idx = 1
        for g in "ABCDEFG":
            for n in range(1,6):
                self.subgrade_map[f'{g}{n}'] = idx
                idx += 1
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all preprocessing steps to input data"""
        df = df.copy()
        
        # Fill missing values
        if 'emp_title' in df.columns:
            df['emp_title'] = df['emp_title'].fillna('others')
        
        # Feature engineering
        df['emp_length_ft'] = np.where(df['emp_length'].isin(self.emp_length_short_term), 0, 1)
        df['address_state_ft'] = df['address_state'].map(df['address_state'].value_counts())
        df['purpose_ft'] = df['purpose'].map(df['purpose'].value_counts())
        df['grade_ft'] = df['grade'].map(self.grade_map)
        df['sub_grade_ft'] = df['sub_grade'].map(self.subgrade_map)
        
        # One-hot encoding
        ohe_cols = ['term', 'verification_status', 'home_ownership']
        for col in ohe_cols:
            df_ohe = pd.get_dummies(df[col], prefix=col).astype(int)
            df = pd.concat([df, df_ohe], axis=1)
        
        # Income and payment features
        df['monthly_income'] = df['annual_income'] / 12
        df['payment_to_income_ratio'] = df['installment'] / df['monthly_income']
        df['loan_to_income_ratio'] = df['loan_amount'] / df['annual_income']
        df['residual_income'] = df['monthly_income'] - df['installment']
        df['is_new_credit_user'] = (df['total_acc'] < 5).astype(int)
        df['log_total_acc'] = np.log1p(df['total_acc'])
        
        df['term_months'] = df['term'].str.extract('(\d+)').astype(int)
        df['interest_burden_ratio'] = df['installment'] * df['term_months'] / df['loan_amount']
        
        return df

    def bin_and_add_feature(self, df: pd.DataFrame, column_name: str, num_bins: int = 3, labels: List[str] = None) -> pd.DataFrame:
        """Bins a numerical column into quantiles and adds it as a new categorical feature"""
        if labels is None:
            labels = [f'{column_name}_Low', f'{column_name}_Medium', f'{column_name}_High']

        unique_values = df[column_name].nunique()
        actual_num_bins = min(num_bins, unique_values)

        if actual_num_bins < 2:
            df[f'{column_name}_binned'] = df[column_name].apply(lambda x: labels[0])
        else:
            df[f'{column_name}_binned'] = pd.qcut(
                df[column_name],
                q=actual_num_bins,
                labels=labels[:actual_num_bins],
                duplicates='drop'
            )
        return df
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced binned and categorical features"""
        # Define columns to bin and their labels
        features_to_bin = {
            'payment_to_income_ratio': ['PTI_Low', 'PTI_Medium', 'PTI_High'],
            'loan_to_income_ratio': ['LTI_Low', 'LTI_Medium', 'LTI_High'],
            'residual_income': ['ResInc_Low', 'ResInc_Medium', 'ResInc_High'],
            'interest_burden_ratio': ['IntBur_Low', 'IntBur_Medium', 'IntBur_High']
        }

        for feature, labels in features_to_bin.items():
            df = self.bin_and_add_feature(df, feature, num_bins=3, labels=labels)

        binned_features = [
            'payment_to_income_ratio_binned',
            'loan_to_income_ratio_binned',
            'residual_income_binned',
            'interest_burden_ratio_binned'
        ]

        df['risk_profile'] = df[binned_features].astype(str).agg('_'.join, axis=1)
        
        # Income brackets
        df['annual_inc_factor'] = pd.cut(df['annual_income'], 100)
        df['annual_income:<25k'] = np.where((df['annual_income'] <= 25000), 1, 0)
        df['annual_income:25k-89k'] = np.where((df['annual_income'] > 25000) & (df['annual_income'] <= 89000), 1, 0)
        df['annual_income:89k-153k'] = np.where((df['annual_income'] > 89000) & (df['annual_income'] <= 153000), 1, 0)
        df['annual_income:153k-207k'] = np.where((df['annual_income'] > 153000) & (df['annual_income'] <= 207000), 1, 0)
        df['annual_income:207k-260k'] = np.where((df['annual_income'] > 207000) & (df['annual_income'] <= 260000), 1, 0)
        df['annual_income:260k-326k'] = np.where((df['annual_income'] > 260000) & (df['annual_income'] <= 326000), 1, 0)
        df['annual_income:>326k'] = np.where((df['annual_income'] > 326000), 1, 0)

        # Interest rate brackets
        df['int_rate_factor'] = pd.cut(df['int_rate'], 30)
        df['int_rate:<0.07'] = np.where((df['int_rate'] <= 0.07), 1, 0)
        df['int_rate:0.07-0.098'] = np.where((df['int_rate'] > 0.07) & (df['int_rate'] <= 0.098), 1, 0)
        df['int_rate:0.098-0.14'] = np.where((df['int_rate'] > 0.098) & (df['int_rate'] <= 0.14), 1, 0)
        df['int_rate:0.14-0.18'] = np.where((df['int_rate'] > 0.14) & (df['int_rate'] <= 0.18), 1, 0)
        df['int_rate:0.18-0.22'] = np.where((df['int_rate'] > 0.18) & (df['int_rate'] <= 0.22), 1, 0)
        df['int_rate:>0.22'] = np.where((df['int_rate'] > 0.22), 1, 0)

        # DTI brackets
        df['dti_factor'] = pd.cut(df['dti'], 50)
        df['dti:<0.06'] = np.where((df['dti'] <= 0.06), 1, 0)
        df['dti:0.06-0.10'] = np.where((df['dti'] > 0.06) & (df['dti'] <= 0.10), 1, 0)
        df['dti:0.10-0.16'] = np.where((df['dti'] > 0.10) & (df['dti'] <= 0.16), 1, 0)
        df['dti:0.16-0.19'] = np.where((df['dti'] > 0.16) & (df['dti'] <= 0.19), 1, 0)
        df['dti:0.19-0.24'] = np.where((df['dti'] > 0.19) & (df['dti'] <= 0.24), 1, 0)
        df['dti:>0.24'] = np.where((df['dti'] > 0.24), 1, 0)

        # Loan amount brackets
        df['loan_amount_factor'] = pd.cut(df['loan_amount'], 50)
        df['loan_amount:<4600'] = np.where((df['loan_amount'] <= 4600), 1, 0)
        df['loan_amount:4600-8k'] = np.where((df['loan_amount'] > 4600) & (df['loan_amount'] <= 8000), 1, 0)
        df['loan_amount:8k-14k'] = np.where((df['loan_amount'] > 8000) & (df['loan_amount'] <= 14000), 1, 0)
        df['loan_amount:14k-20k'] = np.where((df['loan_amount'] > 14000) & (df['loan_amount'] <= 20000), 1, 0)
        df['loan_amount:20k-25k'] = np.where((df['loan_amount'] > 20000) & (df['loan_amount'] <= 25000), 1, 0)
        df['loan_amount:25k-29k'] = np.where((df['loan_amount'] > 25000) & (df['loan_amount'] <= 29000), 1, 0)
        df['loan_amount:>29k'] = np.where((df['loan_amount'] > 29000), 1, 0)

        # One-hot encode binned features
        cat_col_to_encode = ['payment_to_income_ratio_binned', 'loan_to_income_ratio_binned', 
                           'residual_income_binned','interest_burden_ratio_binned']

        for col in cat_col_to_encode:
            df_ohe = pd.get_dummies(df[col], prefix=col).astype(int)
            df = pd.concat([df, df_ohe], axis=1)

        df['risk_profile'] = df['risk_profile'].map(df['risk_profile'].value_counts())
        
        return df
    
    def clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop unnecessary columns and clean up features"""
        cols_to_drop = ['address_state', 'emp_length', 'grade', 'home_ownership', 'purpose',
                       'sub_grade', 'term', 'verification_status', 'total_acc',
                       'annual_income', 'dti', 'installment', 'term_months',
                       'payment_to_income_ratio_binned', 'loan_to_income_ratio_binned', 
                       'residual_income_binned', 'interest_burden_ratio_binned', 'risk_profile', 
                       'annual_inc_factor', 'int_rate_factor', 'dti_factor', 'loan_amount_factor']

        noisy_cols_from_onehotencoding = ['term_ 60 months', 'verification_status_Source Verified', 
                                         'home_ownership_OTHER', 'annual_income:>326k', 'int_rate:<0.07', 
                                         'dti:<0.06', 'loan_amount:<4600']
        
        # Drop columns that exist
        existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        existing_noisy_cols = [col for col in noisy_cols_from_onehotencoding if col in df.columns]
        
        df = df.drop(existing_cols_to_drop + existing_noisy_cols, axis=1)
        
        return df
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main prediction function"""
        self.logger.info("Starting prediction process")
        
        try:
            # Convert input to DataFrame
            df = pd.DataFrame([input_data])
            self.logger.debug(f"Input data shape: {df.shape}")
            
            # Apply preprocessing pipeline
            df = self.preprocess_data(df)
            self.logger.debug(f"After preprocessing: {df.shape}")
            
            df = self.create_advanced_features(df)
            self.logger.debug(f"After feature creation: {df.shape}")
            
            df = self.clean_features(df)
            self.logger.debug(f"After cleaning: {df.shape}")
            
            # Align features with training data
            if self.expected_features is not None:
                missing_cols = [col for col in self.expected_features if col not in df.columns]
                if missing_cols:
                    self.logger.debug(f"Adding {len(missing_cols)} missing columns")
                    for col in missing_cols:
                        df[col] = 0
                
                # Reorder columns to match training
                df = df[self.expected_features]
                self.logger.debug(f"Final feature alignment complete: {df.shape}")
            
            # Make prediction
            prediction = self.model.predict(df)[0]
            probability = self.model.predict_proba(df)[0]
            
            result = {
                'prediction': int(prediction),
                'probability_good_loan': float(probability[0]),
                'probability_bad_loan': float(probability[1]),
                'loan_decision': 'APPROVED' if prediction == 0 else 'REJECTED',
                'risk_level': 'High' if probability[1] > 0.7 else 'Medium' if probability[1] > 0.3 else 'Low'
            }
            
            self.logger.info(f"Prediction successful: {result['loan_decision']} (risk: {result['risk_level']})")
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise

# Usage example
if __name__ == "__main__":
    # Initialize pipeline with feature names
    pipeline = LoanApprovalInferencePipeline(
        r'model/best_rf.pkl',
        r'model/feature_names.pkl'
    )
    
    # Example input
    sample_input = {
        'annual_income': 50000,
        'dti': 0.15,
        'loan_amount': 10000,
        'int_rate': 0.12,
        'installment': 300,
        'grade': 'B',
        'sub_grade': 'B2',
        'emp_length': '5 years',
        'home_ownership': 'RENT',
        'verification_status': 'Verified',
        'purpose': 'debt_consolidation',
        'term': '36 months',
        'total_acc': 15,
        'address_state': 'CA'
    }
    
    # Make prediction
    result = pipeline.predict(sample_input)
    print(result)