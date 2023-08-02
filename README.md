


# ✅ PROCESS

To create the dashboard in Power BI, the company provides you some data sets in the drive and the leadership wants that you with your dashboard helps to resolve the questions:

1 - Does the accuracy decreases in the last years as sales team mentioned?

2 - Does the accuracy behavior changes based on flavor and size?

3 - Company wants to expand this analysis to the other categories but is concerned about times to refresh and get results in Power Bi. What do you suggest implementing to avoid a big latency to visualize results?

4 - Which data model do you use to create this Power BI and Why do you select that data model?

In the beginning, I went through the entire content of the PDF and examined the datasets stored on the drive. It became evident that these datasets contained a wealth of information organized in multiple rows and columns. Consequently, I conducted an analysis of all the variables present in the columns across all the datasets. Subsequently, I began observing the relationships between these variables among the five datasets. To facilitate this analysis, I imported all the datasets into PowerBI. Although PowerBI is an excellent tool, I decided to optimize the Data Exploration phase by utilizing the programming language "R". I made this choice because R is a highly robust tool and I have a strong proficiency in the language. This shift allowed me to delve deeper into solving the business problem at hand.

![Screenshot 2023-06-29 at 14 38 09](https://github.com/lucashomuniz/Project-10/assets/123151332/690c7ec2-aaf9-4103-a2f9-7f0bcd56b54b)

# ✅ ANSWER QUESTION 1

First, I imported the datasets into “R” and started to develop an Exploratory Analysis. Then, I made some changes to the column names of ACTUALS and FORECASTS datasets. Based on the premise that the variable SKU_ACTIVO and PRODUCT_ID are similar, the column has been renamed. After that, the PERIOD column was transformed, in order to create two other columns, YEAR and MONTH. As mentioned in the PDF file “The data will be based on the chocolates category, however the datasets will contain records from other categories, which should not be considered in the analysis”. Based on this information, I calculated the amount of PRODUCT_ID with the CHOCOLATE category within the MASTER_PRODUCT dataset and the result was 350 lines. Using this premise as a basis to develop the analysis, two new data frames were created: NEW_ACTUALS and NEW_FORECASTS.

The dataframe NEW_ACTUALS was created by merging the ACTUALS dataset with the ACTIVE_PRODUCT dataset based on the BCODE column. Then, this resulted dataset was merged again with the MASTER_PRODUCT dataset based on the PRODUCT_ID column. Finally, this last resulted dataset was filtered based on the PRODUCT_ID column, to show only those with the CHOCOLATE category.
Sequentially, in a similar way, to create the NEW_FORECASTS dataframe, first the FORECASTS dataset was merged with the MASTER_PRODUCT dataset based on the PRODUCT_ID column. Later a filter was created so that the new dataframe only presents the category that have CHOCOLATE on it. From that, the next step was to join these two dataframes (NEW_ACTUALS and NEW_FORECASTS) generating the MERGE dataframe. Therefore, the columns that were generated with NA values were replaced by zero and some columns that would not be used were also excluded (PERIOD, MONTH and HOLDING). 

Following through the process of data cleaning and transformation, I successfully created the S.F.A. (Sales Forecast Accuracy) measure using data from the years 2023, 2022, 2021, and 2020. The S.F.A. measure served as a valuable tool for assessing the accuracy of sales forecasts. Within PowerBI, on PAGE 2, I presented a graphical representation illustrating the close alignment between the actual tons sold and the predicted tons. The graph demonstrated that the values of tons sold closely matched the values of tons predicted, with a minimal difference ranging between two to five thousand tons. This indicates a high level of accuracy in the sales forecasting process. The visual representation provided a clear visualization of the alignment between predicted and actual sales, confirming the effectiveness of the forecasting model.

![Screenshot 2023-06-29 at 14 24 44](https://github.com/lucashomuniz/Project-10/assets/123151332/2ec17422-4c94-434a-9ddb-1e5dee5c78fe)

Moving on to PAGE 3 of the PowerBI report, a detailed analysis of the measure's accuracy over time can be observed. The report highlights that the peak accuracy of the measure was achieved in the year 2020, which could possibly be attributed to its initial development and refinement during that period. Following that, in the subsequent two years, the measure maintained a consistent level of accuracy without any significant variations. However, in the current year, there is a noticeable drop in accuracy. It is important to note that this decline might be exaggerated due to the limited availability of data from only the first few months of 2023. Considering this factor, it is reasonable to believe that the sales team's concerns about the drop in accuracy are valid. Nonetheless, it is worth acknowledging that even though there has been a gradual decrease in accuracy over the years, the measure still maintains a mathematically high value. This suggests that the forecasting model is relatively reliable despite the observed decline. The overall accuracy remains at a commendable level, indicating that the sales forecasting process is still valuable for providing insights and predictions for decision-making within the organization.

![Screenshot 2023-06-29 at 14 26 04](https://github.com/lucashomuniz/Project-10/assets/123151332/8a5d7b2a-460f-4b05-9f4b-84a95f496cf5)

# ✅ ANSWER QUESTION 2

To address both the second and first questions, I leveraged the power of the R programming language to calculate the S.F.A. measure. Utilizing the MERGE dataframe, I filtered the quantities of tons sold and tons forecasted for each specific size and flavor. This allowed for a more granular analysis of accuracy based on these attributes. On PAGE 4 of the report, I presented graphical representations that provide insights into the relationship between accuracy and size as well as accuracy and flavor. When examining size, the analysis reveals that the S.F.A. measure exhibits the highest accuracy in the "medium" size category, while the accuracy remains relatively similar across the other sizes. In terms of flavor, it is noteworthy that all flavors demonstrate a remarkably high measure of accuracy, with the exception of the "WITH MILK" flavor, which exhibits a lower accuracy. Among all the flavors, the "MILK" flavor stands out as having the highest accuracy. By comparing the S.F.A. measure across different sizes and flavors, it becomes evident that the accuracy of the measure varies depending on these specific attributes. This suggests that certain sizes and flavors have a significant impact on the accuracy of the sales forecasting process. This insight can be invaluable for decision-making and strategic planning, as it allows the sales team to focus their efforts on optimizing accuracy for specific sizes and flavors where improvements may be needed.

![Screenshot 2023-06-29 at 14 27 37](https://github.com/lucashomuniz/Project-10/assets/123151332/29e25031-3a5a-493d-b966-d37b351f62a6)

# ✅ ANSWER QUESTION 3

As you could see, in order to answer this test, other than PowerBI I used de programming language R. I consider R to be a powerful tool for tackling statistical analysis problems. One recommendation I have is to incorporate programming languages like R or Python to develop algorithms and calculate statistical insights. These insights can then be saved in a .csv file and imported into PowerBI, reducing processing time. Another suggestion for minimizing latency and enhancing performance when expanding the analysis to other categories within Power BI is to consider the following strategies:

Data Aggregation: Instead of loading and processing the entire raw data for each category, consider pre-aggregating the data at a higher level, such as daily, weekly, or monthly. This summarization reduces the overall data volume and speeds up processing and visualization.

Incremental Data Refresh: If the underlying data undergoes frequent updates, implementing an incremental data refresh strategy can be beneficial. Rather than refreshing the entire dataset, only load and process the new or changed data since the last refresh. This approach significantly reduces processing time and enables faster updates for visualizations.

By implementing these strategies, you can minimize latency and optimize the performance of refreshing and visualizing results in Power BI when expanding the analysis to other categories. However, it's crucial to consider the specific requirements, data volumes, and infrastructure limitations of your environment to determine the most suitable approach.

# ✅ ANSWER QUESTION 4

Essentially, when approaching the modeling process, I adopted a perspective of placing myself in the shoes of the audience, imagining how someone without technical expertise would perceive the visualizations. My objective was to create data visualizations that even non-technical individuals could understand and derive a minimum level of information from. Using the provided Tabular data models as a foundation, I aimed to construct an intuitive structure that would align well with visualizations such as Column Plots and Line Charts. Given that the data was organized in tables with rows and columns, it facilitated the representation of hierarchical relationships and numerical values in a straightforward manner. I am aware that Tabular data models in Power BI employ in-memory processing, where the data is loaded and stored in a compressed columnar format in memory. Even if the data was not originally in a tabular model, I would still transform it into one. This approach ensures faster query response times and facilitates interactive and responsive visualizations. Additionally, in PowerBI, I utilized the DAX (Data Analysis Expressions) language to develop the S.F.A. measure. DAX provides a powerful and flexible range of functions for creating calculations and measures.

It is important to emphasize that establishing relationships between tables was crucial for the analysis. Without these connections, I would not have been able to conduct the analysis effectively. For instance, in pursuit of the chocolate category as the goal, I linked the FORECAST dataset with the MASTER_PRODUCT dataset using the PRODUCT_ID (first relation). Subsequently, I had to link the ACTUALS dataset with the variable PRODUCT_ID, even though it did not possess that specific column. Instead, it had a BCODE column. Thus, I established the second relation by merging the ACTUALS dataset with the ACTIVE_PRODUCT_ID dataset. This enabled me to construct the third and final relation by merging the combined ACTUALS and ACTIVE_PRODUCT_ID dataset with the MASTER_PRODUCT dataset. Through these linkages, I was able to develop a solution. The underlying principle guiding this process was the use of a "logical data model."


# ✅ DATA SOURCES

https://drive.google.com/drive/folders/1b1mEm8qayzHRh9SAhvUGzNxS61okNh9R?usp=share_link

https://1drv.ms/u/s!ApbaJdNoYMtWh9dO9fD2iFqRYbzABw?e=pAqLGf
