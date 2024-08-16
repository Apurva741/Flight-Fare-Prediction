# Flight-Fare-Prediction

 Flight Fare Prediction Project

Overview:
This machine learning project aims to predict flight fares based on various features like airline, source, destination, stops, and timing information. The project demonstrates proficiency in data analysis, feature engineering, and machine learning model development and optimization.

Key Technologies and Libraries:
- Python
- Pandas for data manipulation 
- NumPy for numerical operations
- Matplotlib and Seaborn for data visualization
- Scikit-learn for machine learning algorithms and tools

Data Analysis:
- Dataset size: 10,683 training samples, 2,671 test samples
- Features analyzed: Airline, Source, Destination, Stops, Date/Time information
- Visualizations created to analyze relationships between features and price

Feature Engineering:
- Date/time features extracted from raw data
- Categorical variables encoded using Label Encoding
- Route information parsed into separate columns
- Null values handled appropriately

Machine Learning Models:
1. Random Forest Regressor
   - Initial model performance:
     * R-squared score on training data: 0.948
     * R-squared score on test data: 0.621
     * Mean Absolute Error: 1527.09
     * Root Mean Squared Error: 2510.89

2. Optimized Random Forest Regressor
   - Hyperparameter tuning performed using RandomizedSearchCV
   - Best parameters:
     * n_estimators: 700
     * max_depth: 15
     * min_samples_split: 3
     * min_samples_leaf: 1
     * max_features: 'sqrt'
   - Optimized model performance:
     * Mean Absolute Error: 1554.39
     * Root Mean Squared Error: 2462.29

Feature Importance:
- Utilized ExtraTreesRegressor to identify most important features
- Top features: Total Stops, Journey Day, Airline

Model Evaluation:
- Residual analysis performed using distplot
- Scatter plots used to visualize predicted vs actual prices

Key Achievements:
1. Successful implementation of end-to-end machine learning pipeline
2. Effective feature engineering to extract meaningful information
3. Hyperparameter tuning to optimize model performance
4. Clear visualization of results and model performance

Areas for Potential Improvement:
1. Explore additional feature engineering techniques
2. Test other machine learning algorithms (e.g. Gradient Boosting, Neural Networks)
3. Collect more data to potentially improve model generalization

This project demonstrates strong skills in data analysis, feature engineering, and machine learning model development using popular Python libraries and tools. The methodical approach to model optimization and evaluation showcases the ability to tackle real-world machine learning problems effectively.
