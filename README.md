# Clustering-Countries-for-Strategic-Aid-Allocation

**Problem statement:**
HELP International is an international humanitarian NGO that is committed to fighting poverty and providing the people of backward countries with basic amenities and relief during the time of disasters and natural calamities.

HELP International have been able to raise around $ 10 million. Now the CEO of the NGO needs to decide how to use this money strategically and effectively. So, CEO has to make decision to choose the countries that are in the direst need of aid. Hence, your Job as a Data scientist is to categorise the countries using some socio-economic and health factors that determine the overall development of the country. Then you need to suggest the countries which the CEO needs to focus on the most.

**Need and Use of Solving the Problem Statement**

**Need**

*Strategic Resource Allocation:*

HELP International aims to maximize the impact of its $10 million funds by strategically allocating aid to countries most in need. Determining which countries require assistance the most is crucial for optimizing the utilization of resources and effectively combating poverty.


*Data-Driven Decision Making:*

Traditional methods of aid allocation may lack precision and objectivity. Leveraging data science techniques allows for a more systematic and evidence-based approach to identifying countries in dire need, ensuring that resources are allocated where they can make the most significant impact.

*Targeted Intervention:*


By categorizing countries based on socio-economic and health factors, HELP International can tailor interventions to address specific challenges faced by different regions. This targeted approach enhances the effectiveness and efficiency of aid delivery, leading to better outcomes for the communities served.

**Use**

*Country Categorization:*

Utilizing socio-economic and health factors, data scientists will categorize countries into distinct groups based on their level of development and need for aid. This segmentation enables HELP International to prioritize assistance to countries facing the most severe challenges.


*Strategic Decision Support:*

The data-driven insights provided by the analysis will inform the CEO's decision-making process regarding aid allocation. By highlighting countries with the greatest need, data science empowers HELP International to make informed, strategic decisions that maximize the impact of available resources.


*Resource Optimization:*

By focusing aid efforts on countries identified as being in the direst need, HELP International can optimize the allocation of its $10 million funds. This ensures that resources are directed where they can have the most significant positive impact on poverty alleviation and community development.



**Benefits**


*Maximized Impact:*

By targeting aid to countries with the greatest need, HELP International can maximize the impact of its resources, effectively addressing poverty and improving living conditions for vulnerable populations.



*Efficient Resource Utilization:*

Data-driven aid allocation enables HELP International to optimize the utilization of its funds, ensuring that every dollar is directed toward initiatives that have the greatest potential for positive change.

*Improved Accountability:*


Using objective criteria to prioritize aid allocation enhances transparency and accountability in decision-making. This helps build trust with stakeholders and donors by demonstrating a commitment to evidence-based practices and results-driven outcomes.

---

This is an **Unsupervised Machine Learning** project focusing on various Clustering methods like KMeans, Hierarchical Clustering and DBSCAN algorithms. The colab notebook in the repository consists of all the **EDA (Exploratory Data Analysis)** and **Hypothesis Testing** performed along with the **Model Building**.


Generated insights to prioritize aid for least-developed countries with high child mortality and economic fragility, using **PCA** and **t-SNE** for cluster visualization.
The clustering analysis revealed significant overlap among data points when using 3 or 6 clusters, with many points classified as noise in the case of DBSCAN. Below is a summary of model performance:

---

### **Model Comparison:**

**KMeans:**

* Interpretability: High
* Compactness (Silhouette Score): 0.260
* Noise Handling: Poor


**Hierarchical Clustering:**

* Interpretability: Moderate
* Compactness: 0.239
* Noise Handling: Poor


**DBSCAN:**
* Interpretability: Low
* Compactness: 0.119
* Noise Handling: Good


Considering 3 clusters as the optimal value (as 2 clusters oversimplify the data structure), KMeans emerges as the most suitable model. It offers high interpretability and the best compactness compared to other models, although it lacks robustness in handling noise.

To finalize the clustering process, the labels from the KMeans model with 3 clusters were attached to a refined dataset. This dataset consists of five key features selected to avoid multicollinearity (down from nine features). It was prepared using raw data without outliers and with Box-Cox transformations applied.

**Deployment:**
The final dataset enables users to input values for these five features to generate clustering results, making it efficient for practical applications while maintaining data integrity and interpretability.

---

## **Overall Analysis and Insights:**

**1. Cluster Characteristics and Their Implications:**

*Highly Developed Countries (Cluster 0):*

Strong economic indicators like high GDP per capita and robust health investments.
Low child mortality and stable inflation reflect overall social and economic stability.
Exports are significantly high, contributing to economic sustainability. These countries represent the least need for external aid or interventions.



*Moderately Developed Countries (Cluster 2):*

These countries show moderate GDP and child mortality rates, but they still face some challenges, particularly with inflation and health investment.
Targeted aid and reforms could help stabilize these economies and improve their developmental trajectory.


*Least Developed Countries (Cluster 1):*

Characterized by low GDP per capita, extremely high child mortality, and volatile inflation.
Low exports and limited health investments reflect economic fragility and social challenges.
These countries require urgent and substantial aid to address basic health, education, and infrastructure needs.


**2. Key Observations on Factors:**

*Exports:*

Countries with high exports (especially in Cluster 0) have stronger economic stability. Policies promoting trade agreements and industrialization could benefit lower-exporting countries.


*Health Investments:*

A direct correlation is observed between health investment and child mortality. Increasing health budgets could significantly reduce mortality rates in developing and least-developed countries.


*Inflation:*

High inflation in clusters 1 and 2 points to economic instability. Financial policies aimed at stabilizing inflation can drive better economic outcomes.


*Child Mortality:*

This is a key metric for assessing development levels. The major differences between clusters highlight where international aid and intervention should prioritize.


### **Recommendations:**

**For Policy Makers:**

*Focus on Basic Needs in Cluster 1 Countries:*

Increase health investments to address high child mortality rates. Implement measures to stabilize inflation and encourage economic activity through microfinance and entrepreneurship programs.
Strengthen export-oriented sectors by providing incentives for industrial and agricultural productivity.


*Boost Economic Development in Cluster 2 Countries:*

Encourage foreign direct investment (FDI) and partnerships to build infrastructure and industries. Introduce targeted healthcare initiatives to address moderate child mortality rates. Focus on education and
skill development to improve workforce productivity.


*Sustain Progress in Cluster 0 Countries:*

Promote innovation and research to maintain competitive advantages. Strengthen resilience to potential economic shocks through diversification of exports and robust fiscal policies.


**For International Organizations and NGOs:**

*Aid Allocation:*

Prioritize Cluster 1 countries for financial aid and development projects, especially in healthcare and infrastructure. Encourage global partnerships to mobilize resources for addressing extreme poverty and
high mortality rates.


*Monitor Economic Indicators:*

Track GDP per capita, child mortality, and inflation trends to identify early signs of improvement or regression in targeted countries.


### **Potential Developments:**

*Regional Collaboration:*

Encourage regional trade agreements and economic unions to strengthen exports in Clusters 1 and 2. Collaborative health programs across neighboring countries can drive efficiency in resource allocation
and service delivery.


*Innovative Financing:*

Introduce impact bonds or other innovative financing mechanisms to drive social outcomes in least-developed countries.


*Technology-Driven Solutions:*

Use data analytics and AI for efficient resource allocation and monitoring of aid effectiveness. Promote digital education and e-health platforms to bridge gaps in education and healthcare services.


*Sustainability Initiatives:*

Focus on renewable energy investments in Clusters 1 and 2 to address energy needs and create jobs. Encourage sustainable agricultural practices to improve exports and food security.


### **Conclusion:**

The analysis shows the importance of tailored interventions based on the development levels of different countries. While highly developed nations can sustain their progress independently, moderately and
least-developed countries require targeted international support to address fundamental issues like health, inflation, and economic growth. By prioritizing these areas, we can bridge the gap in global
development and promote a more equitable future.




A Flask API is built for model prediction purpose and also a simple Streamlit app is built for the same and deployed for users to try on!  ---> https://country-status-predict.streamlit.app/

Tableau Dashboard for data analysis and visualizations can be found at ---> https://public.tableau.com/app/profile/pritam.kandula/viz/Clustering_Countries_portfolio/GlobalHealthandEconomicDisparities


