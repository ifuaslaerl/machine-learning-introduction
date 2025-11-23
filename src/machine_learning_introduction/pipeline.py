import typing
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

class Pipeline:
    def __init__(self, 
                 scaler: typing.Optional[TransformerMixin], 
                 fill_categorical_strategy: typing.Callable[[pd.Series], any], 
                 fill_numerical_strategy: typing.Callable[[pd.Series], float],
                 exclude_collumns: typing.List[str],
                 model: BaseEstimator,
                 y_column: str,
                 identifier: str,
                 test_size: float = 0.1):
        self.scaler = scaler
        self.fill_categorical_strategy = fill_categorical_strategy
        self.fill_numerical_strategy = fill_numerical_strategy
        self.exclude_collumns = exclude_collumns
        self.model = model
        self.y_column = y_column
        self.identifier = identifier
        self.test_size = test_size
        
        # Dictionary to store learned imputation values from training data
        self.imputation_values = {} 
    
    def fill_categorical(self, df: pd.DataFrame, fit: bool = False):
        """
        Fills categorical values.
        If fit=True, calculates the fill value from the current df (training) and stores it.
        If fit=False, uses the stored value to fill.
        """
        for col in df.select_dtypes(include=['object']).columns:
            if col != self.identifier and col != self.y_column:
                if fit:
                    # Learn the value (e.g., Mode) from training data
                    fill_value = self.fill_categorical_strategy(df[col])
                    self.imputation_values[col] = fill_value
                else:
                    # Use the value learned from training
                    fill_value = self.imputation_values.get(col)
                    
                    # Fallback if column wasn't seen in training (rare)
                    if fill_value is None: 
                        fill_value = self.fill_categorical_strategy(df[col])

                df[col] = df[col].fillna(fill_value)
        return df

    def encode_categorical(self, df: pd.DataFrame):
        """Converts categorical variables to dummies."""
        cols_to_encode = [col for col in df.select_dtypes(include=['object']).columns 
                          if col != self.identifier and col != self.y_column]
        
        if not cols_to_encode:
            return df

        df_dummies = pd.get_dummies(df[cols_to_encode], drop_first=True)
        df = df.drop(columns=cols_to_encode).join(df_dummies)
        return df
    
    def fill_numerical(self, df: pd.DataFrame, fit: bool = False):
        """
        Fills numerical values. 
        Fit=True learns the mean/median from Train. 
        Fit=False applies it to Test.
        """
        cols = df.select_dtypes(include=[np.number]).columns
        cols = [c for c in cols if c != self.identifier and c != self.y_column]

        for col in cols:
            if fit:
                fill_value = self.fill_numerical_strategy(df[col])
                self.imputation_values[col] = fill_value
            else:
                fill_value = self.imputation_values.get(col, 0) # Default to 0 if missing

            df[col] = df[col].fillna(fill_value)
        return df
    
    def normalize(self, df: pd.DataFrame, fit: bool = False):
        """Normalizes data. Fits scaler only if fit=True."""
        if self.scaler is None:
            return df
            
        numerical_cols = df.select_dtypes(include=[np.number, int, float, bool]).columns
        cols_to_scale = [c for c in numerical_cols if c != self.identifier and c != self.y_column]
        
        if cols_to_scale:
            if fit:
                df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
            else:
                df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
            
        return df

    def view_config(self):
        print(f"Model: {self.model}")
        print(f"Scaler: {self.scaler}")
        print(f"Cat Strategy: {self.fill_categorical_strategy.__name__}")
        print(f"Num Strategy: {self.fill_numerical_strategy.__name__}")
        print(f"Excluded: {self.exclude_collumns}")

    def process(self, train: pd.DataFrame, validate: pd.DataFrame, see_action: bool = False):
        """Process training and validation data safely without leakage."""
        
        # Save target and identifier
        y_train = train[self.y_column]
        ids_validate = validate[self.identifier] if self.identifier in validate.columns else None

        # Drop excluded columns
        if self.exclude_collumns:
            train = train.drop(columns=[c for c in self.exclude_collumns if c in train.columns])
            validate = validate.drop(columns=[c for c in self.exclude_collumns if c in validate.columns])
        
        # 1. Fill Categorical (Fit on Train, Transform Validate)
        train = self.fill_categorical(train, fit=True)
        validate = self.fill_categorical(validate, fit=False)
        if see_action: print("Categorical data filled")
        
        # 2. Encode (Dummies)
        train = self.encode_categorical(train)
        validate = self.encode_categorical(validate)
        if see_action: print("Data encoded")
        
        # 3. Fill Numerical (Fit on Train, Transform Validate)
        train = self.fill_numerical(train, fit=True)
        validate = self.fill_numerical(validate, fit=False)
        if see_action: print("Numerical data filled")
        
        # 4. Align columns
        train, validate = train.align(validate, join="left", axis=1, fill_value=0)
        
        # Restore target and identifier
        train[self.y_column] = y_train
        if ids_validate is not None:
            validate[self.identifier] = ids_validate

        if see_action: print("Columns aligned")
        
        # 5. Normalize (Fit on Train, Transform Validate)
        train = self.normalize(train, fit=True)
        validate = self.normalize(validate, fit=False)
        if see_action: print("Data normalized")
        
        # Prepare X and y
        X_train = train.drop(columns=[self.y_column, self.identifier], errors='ignore')
        y_train = train[self.y_column]
        
        X_validate = validate.drop(columns=[self.y_column, self.identifier], errors='ignore')

        # Fit and Predict
        self.model.fit(X_train, y_train)
        prediction = self.model.predict(X_validate)
        
        # Format answer
        answer = pd.DataFrame()
        if ids_validate is not None:
            answer[self.identifier] = ids_validate
        answer[self.y_column] = prediction
        
        return train, answer
