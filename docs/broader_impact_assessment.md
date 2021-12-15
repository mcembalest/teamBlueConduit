## Broader Impact Assessment

We hope our work has a direct impact on health outcomes in Flint and other cities. We have shown that our diffusion model can help cities reduce the amount of time and money required to get lead out of the ground. This will, ultimately, create safer infrastructure and lower monetary burdens for cities and their residents.

However, we also investigated possible negative consequences that could arise from our diffusion model. One possible area of negative impact is disparate impact with respect to demographics such as age, race, and income.

Since our goal is to optimize the cost of replacement city-wide, certain neighborhoods are at risk of having their lead replacement unfairly prioritized at the expense of others. We investigated how our model changed the digging priority order across demographic factors. Specifically, we used Flint census tract data to investigate correlations between demographic variables and the changes in digging order produced by our model. We arrived at the following results:

**1. the available racial demographics of the census tract, calculated as the following: (B_total_pop_black/(B_total_pop_black+B_total_pop_white) %.** Our model seems to have no correlation (Spearman correlation -.03, with low-confidence p-value of 0.84).

**2.  the median age.** Our model has a slight negative correlation (Spearman correlation -0.2, with medium-confidence pvalue of 0.23)

**3. the average value of** **Residential_Building_Value.** Our model has negative correlation (Spearman correlation -0.4, with high-confidence pvalue of 0.01).

Overall, we found little impact with respect to age of residents and density of Black residents in neighborhoods. However, we have evidence to suggest our model prioritizes areas with lower residential value (i.e. lower property value neighborhoods). In other words, our model tends to bump up lower value homes in the dig order.

Note: The data underlying our project represents a city (Flint) with historically embedded racial discrimination in the form of housing disparities and redlining. So, we are not attempting to claim that the use of machine learning in this setting will be globally unbiased with respect to all protected variables. Rather, we are claiming that our diffusion method does not seem to *amplify* any bias on the basis of race or age, compared to BlueConduit’s baseline model.This assessment is specific to just one dataset: the ~27,000 homes in Flint. To improve on this impact assessment in the future, we would investigate how diffusion impacts dig orders in more cities.Another possible area of negative impact for future examination is whether our smoothed probabilities post-diffusion disrupt BlueConduit's longer term goal of having a calibrated model (e.g. of the homes predicted with 70% probability of lead, 70% of them actually end up having lead).