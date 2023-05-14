# Backgrounds
In this project, I will create an XGBoost model classifying employee attritions with employee records, and then apply LIME and SHAP to find actionable insights. The data used to build the classifier is provided by the IBM Data Science team, where each record in the data contains feature values of a corresponding employee, and the target variable Attrition indicates whether the employee has left the company (0: still employed, 1: left).

## Data Description
 * Age: Employees' age  
 * Attrition: Whether or not the employee has left, 'Yes' 'No'  
 * BusinessTravel: Frequency of business travel, 'Non-Travel' 'Travel Rarely' 'Travel Frequently'  
 * Department: Department of employment, 'Human Resources' 'Research & Development' 'Sales'  
 * DistanceFromHome: Commute distance  
 * Education: Highest level of education, {1:'Below College', 2:'College', 3:'Bachelor', 4:'Master', 5:'Doctor'}  
 * EducationField: Field of study, 'Human Resources' 'Life Sciences' 'Marketing' 'Medical' 'Technical Degree' 'Other'  
 * EnvironmentSatisfaction: Satisfaction with work environment, {1:'Low', 2:'Medium', 3:'High', 4:'Very High'}  
 * Gender: Gender, 'Male' 'Female'  
 * JobInvolvement: Level of job involvement, {1:'Low', 2:'Medium', 3:'High', 4:'Very High'}  
 * JobLevel: Job level, 1, 2, 3, 4, 5  
 * JobRole: Job role, 'Healthcare Representative' 'Human Resources' 'Laboratory Technician' 'Manager' 'Manufacturing Director' 'Research Director' 'Research Scientist' 'Sales Executive' 'Sales Representative'  
 * JobSatisfaction: Satisfaction with job performance, {1:'Low', 2:'Medium', 3:'High', 4:'Very High'}  
 * MaritalStatus: Marital status, 'Single' 'Married' 'Divorced'  
 * MonthlyIncome: Monthly income  
 * NumCompaniesWorked: Number of previous employers  
 * OverTime: Whether or not the employee works overtime, 'Yes' 'No'  
 * PercentSalaryHike: Percent increase in salary  
 * PerformanceRating: Job performance rating, {1:'Low', 2:'Good', 3:'Excellent' 4:'Outstanding'}  
 * RelationshipSatisfaction: Satisfaction with work relationships, {1:'Low', 2:'Medium', 3:'High', 4:'Very High'}  
 * StockOptionLevel: Level of stock options, 0, 1, 2, 3  
 * TotalWorkingYears: Total years of work experience  
 * TrainingTimesLastYear: Number of job training sessions attended last year  
 * WorkLifeBalance: Work-life balance, {1:'Bad' 2:'Good' 3:'Better' 4:'Best'}  
 * YearsAtCompany: Number of years with the company  
 * YearsInCurrentRole: Number of years in current position  
 * YearsSinceLastPromotion: Number of years since last promotion  
 * YearsWithCurrManager: Number of years with current manager  

## XAI

### LIME

### Shapley Values
Cooperative games are a branch of game theory where players in a game work together to achieve a common goal. In cooperative games, the possible outcomes of collaboration for each combination of members are measured, and then the marginal contribution and Shapley value of each participant are calculated to determine their contributions in actual teamwork. In machine learning, the concepts of marginal contribution and Shapley value can be applied to measure how features affect the results within a model.

For instance, let's consider a "black box" model with three features: x1, x2, and x3. This model performs predictions for each possible combination of features. The marginal contribution and Shapley value of each feature are measured based on the model's prediction results.

|Model|x1|x2|x3|Outcome|
|---|---|---|---|---|
|m1|False|False|False|28|
|m2|True|False|False|32|
|m3|False|True|False|31|
|m4|False|False|True|30|
|m5|True|True|False|32|
|m6|True|False|True|33|
|m7|False|True|True|32|
|m8|True|True|True|35|

Let's examine the marginal contribution of x1. In a model with only one feature, the difference in results between m2 with x1 included and m1 without x1 is 4, so the marginal contribution of x1 in a single-feature model is 4. Next, there are two dual-feature models that include x1 (m5 and m6). If x1 is removed from m5, the result becomes m3, with a difference of 32-31=1. Similarly, if x1 is removed from m6, the result is different by 3. Finally, removing x1 from the full model m8 also yields a difference of 3. This can be obtained by subtracting the result of m7 from the result of m8. Therefore, the marginal contribution of x1 is as follows:

Single-feature model: m2 - m1 = 4
Dual-feature model: m5 - m3 = 1
Dual-feature model: m6 - m4 = 3
Triple-feature model: m8 - m7 = 3
From each case of the marginal contribution of x1, we can obtain the Shapley value through weighted sum. For example, among the three single-feature models, there is only one case where x1 can be excluded. Therefore, the weight for the marginal contribution of x1 in the single-feature model is 1/3. Similarly, there are a total of three dual-feature models, and in each model, one of the two features can be excluded. In this case, the weight for the marginal contribution of x1 is 1/6, since there are three models and two possible exclusions within each model. Finally, there is only one full model, and there are three possible exclusions within the model. Therefore, the weight is 1 x 1/3. Thus, the Shapley value of feature x1 is as follows:

$$
4 \ast \frac{1}{3} + 1 \ast \frac{1}{6} + 3 \ast \frac{1}{6} + 3 \ast \frac{1}{3}
$$

The definition of Shapley value is as follows, where $\phi$ represents the Shapley value of the $i$-th feature, $S$ represents the set of features excluding the $i$-th feature, and $F$ represents the set of all features.

$$
\phi_i = \sum_{S \subseteq F/\{i\}}\frac{|S|!\ast(|F| - |S| - 1)!}{|F|!} \ast (f(S \cup \{i\}) - f(S))
$$

In this formula, $f(S)$ represents the model's prediction when only considering the features in set $S$, while $f(S \cup {i})$ represents the model's prediction when considering both the $i$-th feature and the features in set $S$. The summation is taken over all possible subsets $S$ of $F$ that do not contain the $i$-th feature. The formula calculates the contribution of the $i$-th feature to the difference between the model's prediction for the entire set of features and the prediction for the subset of features without the $i$-th feature, weighted by the number of ways that the feature can be added to different subsets.

### SHAP

SHAP (SHapley Additive exPlanations) is a framework for interpreting machine learning models based on Shapley values. SHAP estimates the Shapley value of each feature using a kernel-based or tree-based approach and aggregates them to explain the model's output. What distinguishes SHAP from the traditional Shapley value approach is that while the latter measures only which features are important overall for the machine learning model, SHAP also provides local explanations for individual predictions, similar to what is provided by LIME.

In machine learning models, a local explanation refers to explaining an individual prediction made by the model. In other words, local explanations interpret how a specific input record was classified by the model and explain it as a contribution of each feature. In the context of SHAP, local explanations are provided in the form of Shapley values, which indicate how much each feature contributed to a particular prediction for a given input instance. Once the Shapley values of each feature are calculated, SHAP can provide a detailed explanation of how the model predicted a specific input record.

On the other hand, understanding the model's behavior and performance on the entire dataset is referred to as global model insights. SHAP first calculates the feature importance scores (Shapley value scores of each feature) for all records in the dataset through local explanations and then calculates the average of the Shapley values to rank the most important features overall for the model. This can help identify the features that have the biggest impact on the model's output and provide insights into the relationship between the features and the target variable.
