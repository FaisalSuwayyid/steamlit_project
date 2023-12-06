
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import hiplot as hip

# from sklearn import datasets


from PIL import Image



######################################################################



df = pd.read_csv('https://raw.githubusercontent.com/FaisalSuwayyid/CMSE830/main/Mushroom.csv', index_col=0)

st.set_page_config(page_title = 'Classification of Mushrooms to Edible and Poisonous',
                   page_icon = 'bar_chart',
                   layout='wide')


col1, col2, col3 = st.columns([0.2, 0.6, 0.2])
with col2:
    st.title('Classification of Mushrooms to Edible and Poisonous')
    tab_Homepage, tab_Data_Validation, tab_Methodology, tab_Histogram, tab_Decision_Tree, tab_SVM, tab_Naiv_Bayes, tab_Conclusion, tab_bio = st.tabs(["Homepage", "Features Describtion and Study Validation", "Methodology", "Histogram Approach", "Decision Tree", "Support Vector Machine", " Naive Bayes", "Conclusion Remarks", "bio"]) 

with tab_Homepage:
        st.header("Introduction")
        st.markdown("In recent times, there has been a growing concern in both the United States and China regarding the issue of mushroom poisoning. Accidentally consuming poisonous mushrooms can lead to severe health problems and even fatalities (1,2). The period between 1999 and 2016 saw an increase in the number of reported cases of severe poisoning resulting from the consumption of foraged mushrooms, whether for culinary or hallucinogenic purposes (1,3). Notably, poison control centers in the United States received approximately 7,500 reports of poisonous mushroom ingestions annually (1,2). In 2016, there were 1,328 visits to healthcare facilities and 100 hospitalizations linked to accidental ingestion of poisonous mushrooms, as indicated by HCUP data. Among the 556 patients diagnosed with accidental poisonous mushroom ingestion, 48 individuals (8.6%) experienced severe adverse outcomes during the period from 2016 to 2018, based on MarketScan data. It is essential to emphasize that while many cases of mushroom poisoning can be prevented, the consumption of wild mushrooms should be avoided unless an expert has positively identified them. Increased efforts are required regarding public health messaging to raise awareness about the potential dangers of mushroom poisoning. Adverse consequences resulting from ingesting poisonous mushrooms can be attributed to the inability of novice mushroom foragers to distinguish between poisonous and nonpoisonous species (1,3). Accidental diagnoses of mushroom poisoning were more prevalent during the summer and were most frequently reported in the western United States. This pattern may indicate regional variations in the popularity of recreational mushroom foraging or the higher prevalence of the potentially deadly and easily misidentified mushroom species Amanita smithiana, which causes gastrointestinal symptoms followed by acute renal failure in this particular region (1,4). The general public must be aware that poisonous mushrooms may closely resemble nonpoisonous ones, cooking mushrooms does not eliminate or neutralize toxins, and therefore, wild mushrooms should never be consumed without the guidance of an expert (1,5).")
        st.markdown("This study and its practical implementation will introduce a method for categorizing a specific group of mushrooms known as the Agaricus and Lepiota Family into two distinct categories: edible and poisonous. This classification will be based on their anatomical characteristics and other relevant features to enhance public awareness regarding avoiding toxic mushrooms. It is imperative, however, to seek the guidance of a qualified expert before considering the consumption of mushrooms. The application involves the analysis of a dataset comprising 8124 samples, each characterized by 22 descriptive features.")

        st.header("Who This Study is Useful for:")
        st.markdown("This study has a general utility for the public. However, its primary beneficiaries are individuals in the agricultural sector and those interested in consuming mushrooms, such as mushroom hunters. It is especially relevant for individuals who frequently venture into natural environments and anticipate potential encounters with mushrooms.")
        st.header("Significance:")
        st.markdown("According to the data, it is possible to classify the mushrooms in the dataset definitively into edible and poisonous categories because no mushrooms share the same characteristics as both edible and poisonous.")
        
        st.header("References:")
        st.markdown("The dataset is obtained from UCI repository, and is called Mushroom. Here is a link to the \
                    dataset: https://archive.ics.uci.edu/dataset/73/mushroom. Other references are listed below:")
        st.markdown("[1] Gold, J. A., Kiernan, E., Yeh, M., Jackson, B. R., & Benedict, K. (2021). Health care utilization and outcomes associated with accidental poisonous mushroom ingestions—United States, 2016–2018. _Morbidity and Mortality Weekly Report_, _70_(10), 337.")
        st.markdown("[2] Brandenburg WE, Ward KJ. Mushroom poisoning epidemiology in the United States. Mycologia 2018;110:637–41. . 10.1080/00275514.2018.1479561 [[PubMed](https://pubmed.ncbi.nlm.nih.gov/30062915)] [[CrossRef](https://doi.org/10.1080%2F00275514.2018.1479561)] [[Google Scholar](https://scholar.google.com/scholar_lookup?journal=Mycologia&title=Mushroom+poisoning+epidemiology+in+the+United+States.&volume=110&publication_year=2018&pages=637-41&pmid=30062915&doi=10.1080/00275514.2018.1479561&)]")
        st.markdown("[3] Diaz JH. Evolving global epidemiology, syndromic classification, general management, and prevention of unknown mushroom poisonings. Crit Care Med 2005;33:419–26. . 10.1097/01.CCM.0000153530.32162.B7 [[PubMed](https://pubmed.ncbi.nlm.nih.gov/15699848)] [[CrossRef](https://doi.org/10.1097%2F01.CCM.0000153530.32162.B7)] [[Google Scholar](https://scholar.google.com/scholar_lookup?journal=Crit+Care+Med&title=Evolving+global+epidemiology,+syndromic+classification,+general+management,+and+prevention+of+unknown+mushroom+poisonings.&volume=33&publication_year=2005&pages=419-26&pmid=15699848&doi=10.1097/01.CCM.0000153530.32162.B7&)]")
        st.markdown("[4] Tulloss RE, Lindgren JE. _Amanita smithiana:_ taxonomy, distribution, and poisonings. Mycotaxon 1992;45:373–87. [http://www.cybertruffle.org.uk/cyberliber/index.htm](http://www.cybertruffle.org.uk/cyberliber/index.htm). [[Google Scholar](https://scholar.google.com/scholar_lookup?journal=Mycotaxon&title=Amanita+smithiana:+taxonomy,+distribution,+and+poisonings.&volume=45&publication_year=1992&pages=373-87&)]")
        st.markdown("[5] American Association of Poison Control Centers. Food and mushroom poisoning. Arlington, VA: American Association of Poison Control Centers; 2020. [https://aapcc.org/prevention/food-mushroom-tips](https://aapcc.org/prevention/food-mushroom-tips)")

with tab_Data_Validation:
    
        st.header("Features:")
        # image = Image.open('AdobeStock_321292924_Preview copy.jpg')
        # st.image(image, caption='Mushroom Anatomy')
        st.markdown("1. cap-shape:                bell, conical, convex, flat, \
                                  knobbed, sunken.")
        st.markdown('2. cap-surface:              fibrous, grooves, scaly, smooth.')
        st.markdown('3. cap-color:                brown, buff, cinnamon, gray, green \
                                        pink, purple, red, white, yellow.')
        st.markdown("4. bruises?:                 bruises, no.")
        st.markdown("5. odor:                     almond, anise, creosote, fishy, foul,\
                                        musty, none, pungent, spicy.")
        st.markdown("6. gill-attachment:          attached, descending, free, notched.")
        st.markdown("7. gill-spacing:             close, crowded, distant.")
        st.markdown("8. gill-size:                broad, narrow.")
        st.markdown("9. gill-color:               black, brown, buff, chocolate, gray, \
                                        green, orange, pink, purple, red, \
                                        white, yellow.")
        st.markdown("10. stalk-shape:              enlarging, tapering.")
        st.markdown("11. stalk-root:               bulbous, club, cup, equal, \
                                        rhizomorphs, rooted, missing=?. ")
        st.markdown("12. stalk-surface-above-ring: fibrous, scaly, silky, smooth.")
        st.markdown("13. stalk-surface-below-ring: fibrous, scaly, silky, smooth.")
        st.markdown("14. stalk-color-above-ring:   brown, buff, cinnamon, gray, orange, \
                                        pink, red, white, yellow.")
        st.markdown("15. stalk-color-below-ring:   brown, buff, cinnamon, gray, orange, \
                                        pink, red, white, yellow.")
        st.markdown("16. veil-type:                partial, universal.")
        st.markdown("17. veil-color:               brown, orange, white, yellow.")
        st.markdown("18. ring-number:              none, one, two.")
        st.markdown("19. ring-type:                cobwebby, evanescent, flaring, large, \
                                        none, pendant, sheathing, zone")
        st.markdown("20. spore-print-color:        black, brown, buff, chocolate, green, \
                                        orange, purple, white, yellow.")
        st.markdown("21. population:               abundant, clustered, numerous, \
                                        scattered, several, solitary. ")
        st.markdown("22. habitat:                  grasses, leaves, meadows, paths, \
                                        urban, waste, woods.")
        
        st.markdown("Many sources do suggest that odor and spore-print color can be helpful starting points for distinguishing between edible and poisonous mushrooms. However, it is essential to note that the effectiveness of these criteria can vary depending on the specific type and family of mushrooms in question. Different mushroom species may have unique characteristics, and relying solely on odor and spore-print color may only sometimes be sufficient for accurate identification. Therefore, while these criteria can be helpful, it is advisable to consider additional factors and consult with an expert when in doubt.")
        
        ####### Data Preview #########
        st.header("Data Preview")
        st.markdown('#')
        with st.expander('Data Preview'):
            # st.write(df)
            st.dataframe(df)
            # st.dataframe(df,
            #              column_config={
            #                  'Target': st.column_config.NumberColumn(format='%d')
            #              })
        ###############################

        st.markdown('#')
        st.markdown('#')
        st.markdown('Examining the proportion of edible and poisonous mushrooms, we observe that the presence of poisonous mushrooms is nearly equivalent to that of edible ones. This implies that assuming a uniform distribution, the likelihood of encountering a poisonous mushroom is relatively high.')
        st.markdown('#')
        st.markdown('#')
        ####### Data Validation #########
        col1, col2, col3 = st.columns(3)
        with col2:
            st.markdown('# Size of Data')

        option_data_validation = st.selectbox(
        "Select a feature to view its histogram",
        tuple([None, 'percent']),
        index=0,
        placeholder="Select contact method...",
        )
        fig = px.histogram(df, x="target", text_auto=True,
                            histnorm = option_data_validation, histfunc = 'count')
        st.plotly_chart(fig, use_container_width=True)

with tab_Methodology:

        ###############################
        st.markdown("The dataset containing mushroom information is categorical, and the study's objective primarily focuses on classification. As a result, the available methods for analyzing such datasets are primarily limited to classification techniques. Histograms, valuable statistical tools, are employed in this study to elucidate the probability distributions of the samples within the dataset. Specifically, two types of histograms are utilized: one to investigate the distribution of samples within individual features and another to explore the distribution of classes based on pairs of features. Additionally, machine learning approaches, including decision trees, naive Bayes, and support vector machines, will be applied as part of the analysis.")
        
        st.header("Machine Learning Methods:")
        st.subheader("Decision Tree:") 
        st.markdown("Decision Trees are a fundamental machine learning technique used in our study to make decisions based on a series of criteria. In essence, they mimic human decision-making processes by creating a tree-like structure of decisions and possible outcomes. Each internal node of the tree represents a decision point based on a specific feature, and each branch represents a possible outcome or further decision point. Decision Trees are interpretable and can be visualized, making them valuable for understanding the reasoning behind classification or regression decisions. They are particularly useful when dealing with categorical data or data with complex decision boundaries. In our study, we will leverage Decision Trees to classify mushrooms as edible or poisonous based on their characteristics.")
        
        
        st.subheader("Support Vector Machine (SVM):")
        st.markdown("Support Vector Machines are powerful tools for our study's classification task. SVMs are known for their ability to find optimal decision boundaries in high-dimensional spaces. They work by identifying the hyperplane that best separates data points of different classes while maximizing the margin between them. SVMs are particularly effective when dealing with datasets that are not linearly separable, as they can employ various kernel functions to map the data into higher-dimensional spaces where separation becomes possible. In our study, SVMs will play a crucial role in classifying mushrooms based on their features, especially when dealing with intricate decision boundaries.")

        st.subheader("Naive Bayes:")
        st.markdown("Naive Bayes is a probabilistic machine learning algorithm that we will employ in our study to classify mushrooms with efficiency and simplicity. It is based on Bayes' theorem and assumes that features are conditionally independent, which is why it's called 'naive.' Despite this simplifying assumption, Naive Bayes often performs remarkably well in classification tasks, particularly when dealing with text data or situations where real independence is not critical. In our study, we'll use Naive Bayes to calculate the probability of a mushroom being edible or poisonous based on its feature set, making it a valuable addition to our classification methodology.")

        st.header("Scores and Validations:")
        st.subheader("Accuracy:")
        st.markdown("Since the problem is classificational, accuracy, e.g., the percentage of classifying data correctly, is important score in validating methods used in solving the problem.")

        st.subheader("Confusion Matrix:")
        st.markdown("Confusion matrix, specially in binary classification, can be used  to evaluate the performance of a classification algorithm in various ways, and help in calculating several metrics. In mushrooms dataset, poisonous mushrooms are labeled with one, and edible mushrooms are labeled with zero. Therefore, in addition to requiring a high accuracy, algorithms are better to have high false positive rate, and low false negative rate.")
        colconf1, colconf2, colconf3 = st.columns([0.3, 0.5, 0.2])
        with colconf2:
            st.dataframe(pd.DataFrame([["True Neg", "False Pos"],["False Neg","True Pos"]], index = ["Actual Neg 0", "Actual Pos 1"], columns = ["Pred Neg 0", "Pred Pos 1"]))
                    

    
    
    
with tab_Histogram:

        ###############################

        st.markdown('Given that the dataset is predominantly comprised of categorical data, histograms will be the primary statistical technique to be applied. The utilization of histograms for each feature is intended to facilitate the discovery of distinctive characteristics that distinguish between edible and poisonous mushrooms and ascertain the distribution of mushrooms within the chosen feature categories. The overarching objective is to pinpoint features where minimal overlap exists between the classes of edible and poisonous mushrooms. Additionally, it is essential to identify situations where either poisonous or edible mushrooms represent a minority within the common classes, as this can aid in refining the analysis.')
        ############ Histogram count ###############
        
        st.markdown('# The Percentage of each Class of a Feature')
        s = df.columns.tolist()
        s = [ele for ele in s if ele != 'target']
        # s = s.remove('target')
        option_data_histogram = st.selectbox(
        "Select a feature to view its histogram",
        tuple(s),
        index=0,
        placeholder="Select contact method...",
        )


        x1 = df[option_data_histogram].unique().tolist()
        fig2=plt.figure(figsize=(6,4), dpi=250)
        ax=fig2.add_subplot(111)

        sns.histplot(data=df, x=option_data_histogram,
                    hue='target', multiple='stack', stat='percent', ax=ax)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        # ax.spines['bottom'].set_color('#DDDDDD')
        plt.xticks(rotation=45)
        plt.title(f'Percentage of classes in {option_data_histogram}')
        # col1, col2, col3 = st.columns([0.25, 0.5, 0.25])
        # with col2:
        st.pyplot(fig2)



        ###############################

        #### Relative Histogram #####
        st.markdown('#')
        st.markdown('Another approach that can enhance our comprehension of the distribution is to examine the proportion of poisonous and edible mushrooms within the same category within the chosen feature.')
        # relative_histplot(df, option)

        fig = px.histogram(df, x=option_data_histogram,
                            text_auto=True, color = 'target', barnorm = 'percent')
        st.plotly_chart(fig, use_container_width=True)

        ##############################
        st.markdown('#')
        st.markdown('#')
        st.markdown('Upon reviewing the features, it becomes evident that odor, spore-print-color, veil color, and population meet the criteria outlined at the beginning of this section. Odor is a promising starting point as it reveals that most poisonous mushrooms possess an odor, while only a tiny fraction are odorless. Additionally, edible mushrooms with odor exhibit distinct odors compared to their poisonous counterparts.')
        st.markdown('#')
        st.markdown('#')



        ##############################
        # col1, col2, col3 = st.columns([0.0, 1, 0.00])
        # with col2:
        st.markdown('# Study of Distribution of Classes of Pair of Features')
        st.markdown('Examining the distribution of classes within a single feature may only sometimes suffice. Instead, delving into the distribution and probability of two pairs of features can provide further insights into how poisonous and edible mushrooms are distributed, particularly when they share common categories within a feature. The table below presents the probabilities associated with pairs of classes for selected two features. The table on the left displays the probability of edible mushrooms within the context of the pair of classes for selected features. In contrast, the table on the right depicts the corresponding probabilities for poisonous mushrooms. A zero percentage in the left table indicates that all mushrooms with that specific pair of classes are poisonous if any exist, and conversely for the right table.')
        st.markdown('#')
        st.markdown('#')

        s = df.columns.tolist()
        s = [ele for ele in s if ele != 'target']

        option_feature1 = st.selectbox(
        "Feature 1",
        tuple(s),
        index=0,
        placeholder="Select contact method...",
        )
        s = [ele for ele in s if ele != option_feature1]
        option_feature2 = st.selectbox(
        "Feature 2",
        tuple(s),
        index=0,
        placeholder="Select contact method...",
        )

        df_p = df[df['target']=='poisonous']
        df_e = df[df['target']=='edible']

        feature1 = option_feature1
        feature2 = option_feature2

        x = df[[feature1, feature2]].value_counts()
        feature1_classes = df[feature1].unique()
        feature2_classes = df[feature2].unique()



        a = np.zeros((len(feature1_classes), len(feature2_classes)))
        # df[[feature1, feature2]].value_counts()
        for i, labeli in zip(range(len(feature1_classes)), feature1_classes):
            for j, labelj in zip(range(len(feature2_classes)), feature2_classes):
                if labelj in x[labeli].index:
                    a[i][j] = x[labeli][labelj]

        a = a/df.shape[0]*100
        A = pd.DataFrame(a, columns = feature2_classes, index = feature1_classes)

        a_e = np.zeros_like(a)

        x_e = df_e[[feature1, feature2]].value_counts()
        feature1_classes_e = df_e[feature1].unique()

        for i, labeli in zip(range(len(feature1_classes)), feature1_classes):
            for j, labelj in zip(range(len(feature2_classes)), feature2_classes):
                if labeli in feature1_classes_e:
                    if labelj in x_e[labeli].index:
                        a_e[i][j] = x_e[labeli][labelj]
                    
        a_e = a_e/df.shape[0]*100
        A_e = pd.DataFrame(a_e, columns = feature2_classes, index = feature1_classes)

        a_p = a - a_e
        A_p = A - A_e

        with st.expander('Probability Table of the two selected features'):
            # st.write(df)
            st.dataframe(A)
            # st.dataframe(df,
            #              column_config={
            #                  'Target': st.column_config.NumberColumn(format='%d')
            #              })

        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            with st.expander('Probability table of the edible ones'):
            # st.write(df)
                st.dataframe(A_e)
        with col2:
            with st.expander('Probability table of the poisonous ones'):
            # st.write(df)
                st.dataframe(A_p)


        st.markdown('#')
        st.markdown('#')
        st.markdown('Following the experimentation, one can infer that the combination of (odor spore-print-color) is valuable, as it exhibits only one shared pair between edible and poisonous mushrooms with a non-zero probability. In general, using pairs of features substantially diminishes the probability of overlap between the two classes.')





        # ######### Sunburst1 plot ############
        st.markdown('#')
        st.markdown('#')
        st.markdown('To facilitate our comprehension of how three features can categorize mushrooms, namely odor, spore-print-color, and either veil color or population, we employ a sunburst plot, which effectively presents a four-dimensional representation.')
        st.markdown('#')
        st.markdown('#')
        col1, col2, col3 = st.columns(3)
        with col1:
            s = df.columns.tolist()
            s = [ele for ele in s if ele != 'target']

            option_first_sunburst1_feature = st.selectbox(
            "Select first feature for sunburst plot",
            tuple(s),
            index=0,
            placeholder="Select contact method...",
            )
            s = [ele for ele in s if ele != 'target' and \
                ele != option_first_sunburst1_feature]
            option_second_sunburst1_feature = st.selectbox(
            "Select second feature for sunburst plot",
            tuple(s),
            index=0,
            placeholder="Select contact method...",
            )
            s = [ele for ele in s if ele != 'target' and \
                ele != option_first_sunburst1_feature and ele != option_second_sunburst1_feature]
            option_third_sunburst1_feature = st.selectbox(
            "Select third feature for sunburst plot",
            tuple(s),
            index=0,
            placeholder="Select contact method...",
            )
            fig = px.sunburst(df, path=[option_first_sunburst1_feature,
            option_second_sunburst1_feature,
            option_third_sunburst1_feature,
            'target'])
        with col2:
            st.plotly_chart(fig)
            
        st.markdown('#')
        st.markdown('#')
        st.markdown('In summary, our findings suggest that to classify mushrooms as either edible or poisonous, it is adequate to employ a well-selected set of six features. One such combination of six features includes (odor, spore-print-color, stalk-color-above-ring, stalk-surface-below-ring, population, ring-type).')
        st.markdown('#')
        st.markdown('#')
        exp = hip.Experiment.from_dataframe(df[['target', 'ring-type', 'population',\
                     'stalk-color-above-ring', 'stalk-surface-below-ring',\
                          'spore-print-color','odor' ]])
        htmlcomp = exp.to_html()
        st.components.v1.html(htmlcomp,width=700, height=600, scrolling=True)



#############################################

with tab_Decision_Tree:

        ###############################
        st.markdown("Previous histograms have aided in comprehending that the mushroom dataset comprises dimensions less than or equal to six. The classification method is manually derived based on the criteria discussed in the histogram-based approach. However, a decision tree can perform a similar classification using systematic calculations to attain an optimal method for categorizing mushrooms. The performance of the decision tree classifier relies on factors such as the desired tree length and the number of features used in the decision-making process. In this methodology, edible mushrooms are labeled zero (represented by an orange color at the end of a branch).")
        st.markdown("In contrast, poisonous mushrooms are assigned label one (indicated by a blue color at the end of a branch). The initial component within the value array depicted in the generated decision tree figure signifies the count of samples labeled as edible at that level. In contrast, the second component represents the count of samples labeled as poisonous at that level. Beneath the tree figure, several tables illustrate the encoding of classes for each selected feature. For example, X[6] <= 2.5 indicates that if the selected feature 6 has an encoded value less than 2.5, the path followed is to the left; otherwise, it goes to the right.")
        st.markdown("The order of the features is essential in decision tree algorithms. Therefore, one might try several orders to have a good decision tree.")
        st.markdown("According to the previous approach, the histogram, a good choice of features might be (odor, spore-print-color, veil color, stalk-color-above-ring, stalk-surface-below-ring, ring-type, poplutaion). With this choice, even with %50 testing subset, the decision tree performs will with this order.")
        
        from sklearn import tree
        from sklearn.preprocessing import OrdinalEncoder
        encX = OrdinalEncoder()
        ency = OrdinalEncoder()
        allowed_options = [labls for labls in df.columns.tolist() if labls !="target"]
        options = st.multiselect(
        'Select features to use in the decision tree',
        allowed_options,
        allowed_options)
        if len(options)>1:
            max_depth = st.slider('Select maximum depth of the tree', 1, len(options), len(options), 1)
        else:
            max_depth = 1
        test_ratio_DT = st.slider('Select ratio of the test samples', 0.1, 0.5, 0.2, 0.05)
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        st.button("Reset Decision Tree")  
        run_decision_tree_state = st.button('Run Decision Tree')
        if run_decision_tree_state:
            st.write('You selected:', options)
            X = df[options].values
            y = df.iloc[:,0].values.reshape(-1, 1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio_DT, random_state = 42) 
            encX.fit(X_train)
            X_enc = encX.fit_transform(X_train)
            X_test_enc = encX.fit_transform(X_test)
            ency.fit(y_train)
            y_enc = ency.fit_transform(y_train)
            y_test_enc = ency.fit_transform(y_test)
            clf = tree.DecisionTreeClassifier(max_depth = max_depth)
            clf = clf.fit(X_enc, y_enc.flatten())
            y_train_pred_dt = clf.predict(X_enc)
            decision_tree_accuracy_training = 100 - np.sum(np.abs(y_train_pred_dt - y_enc.flatten()))/X_enc.shape[0]*100
            # st.markdown(f"Training accuracy = {decision_tree_accuracy_training}")
            y_test_pred_dt = clf.predict(X_test_enc)
            # st.markdown(f"Testing accuracy = {accuracy_score(y_test_enc, y_test_pred_dt)*100}")
            results = pd.DataFrame([[decision_tree_accuracy_training, accuracy_score(y_test_enc, y_test_pred_dt)*100]], columns = ["Training Accuracy","Testing Accuracy"])
            colr1, colr2, colr3 = st.columns([0.3, 0.5, 0.2])
            with colr2:
                st.dataframe(results)
            fig_tree, tree_axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
            tree.plot_tree(clf,filled=True,rounded=True, ax=tree_axes)
            st.pyplot(fig_tree)
            for i in range(len(options)):
                with st.expander(f'Encoding of {options[i]} classes'):
                #     # st.markdown(encX.categories_[i])
                    st.dataframe(pd.DataFrame(encX.categories_[i]))
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            fig_conf, conf_axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
            ConfusionMatrixDisplay.from_estimator(clf, X_test_enc,y_test_enc, ax = conf_axes)
            colrr1, colrr2, colrr3 = st.columns([0.3, 0.5, 0.2])
            with colrr2:
                st.subheader("Confusion Matrix")
            st.pyplot(fig_conf) 
        else:
            st.write('Please run the decision tree')
        
            
        


with tab_SVM:

        ###############################
        st.markdown("Support vector machine is insensitive to features order, unlike the decision tree. However, it is difficult to make sense out of the support vector mahcine prediction method. To compare it with the decision tree, choosing the same previous collection of features is essential. This approach performs will when %50 split is used. It performs good with %20 however. In both cases, the algorithm has low FP rate, and high FN rate which is undesirable.")
        
        from sklearn import svm
        from sklearn.preprocessing import OrdinalEncoder
        encX3 = OrdinalEncoder()
        ency3 = OrdinalEncoder()
        allowed_options3 = [labls for labls in df.columns.tolist() if labls !="target"]
        options3 = st.multiselect(
        'Select features to use in the support vector machine',
        allowed_options3,
        allowed_options3)
        test_ratio_svm = st.slider('Select ratio of the svm test samples', 0.1, 0.5, 0.2, 0.05)
        st.button("Reset Support Vector Machine")  
        run_svm_state = st.button('Run Support Vector Machine')
        if run_svm_state:
            st.write('You selected:', options3)
            X3 = df[options3].values
            y3 = df.iloc[:,0].values.reshape(-1, 1)
            X_train, X_test, y_train, y_test = train_test_split(X3, y3, test_size=test_ratio_svm, random_state = 42) 
            encX3.fit(X_train)
            X_enc3 = encX3.fit_transform(X_train)
            ency3.fit(y_train)
            y_enc3 = ency3.fit_transform(y_train)
            X_test_enc3 = encX3.fit_transform(X_test)
            y_test_enc3 = ency3.fit_transform(y_test)
            model = svm.SVC()
            model.fit(X_enc3, y_enc3)
            y_train_pred_svm = model.predict(X_enc3)
            svm_train_accuracy = 100 - np.sum(np.abs(y_train_pred_svm - y_enc3.flatten()))/X_enc3.shape[0]*100
            # st.markdown(f"Accuracy = {svm_train_accuracy}")
            y_test_pred_svm = model.predict(X_test_enc3)
            # st.markdown(f"Testing accuracy = {accuracy_score(y_test_enc3, y_test_pred_svm)*100}")
            results = pd.DataFrame([[svm_train_accuracy, accuracy_score(y_test_enc3, y_test_pred_svm)*100]], columns = ["Training Accuracy","Testing Accuracy"])
            colr1, colr2, colr3 = st.columns([0.3, 0.5, 0.2])
            with colr2:
                st.dataframe(results)
            fig_conf_svm, conf_axes_svm = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            ConfusionMatrixDisplay.from_estimator(model, X_test_enc3,y_test_enc3, ax = conf_axes_svm)
            colrrrr1, colrrrr2, colrrrr3 = st.columns([0.3, 0.5, 0.2])
            with colrrrr2:
                st.subheader("Confusion Matrix")
            st.pyplot(fig_conf_svm) 
        else:
            st.write('Please run the support vector machine')
       

with tab_Naiv_Bayes:

        ###############################
        st.markdown("Naive bayes classifier is insensitive to features order, similar to support vector machine. It also has similar performance.")
        
        from sklearn.preprocessing import OrdinalEncoder
        from sklearn.naive_bayes import CategoricalNB

        encX2 = OrdinalEncoder()
        ency2 = OrdinalEncoder()
        allowed_options2 = [labls for labls in df.columns.tolist() if labls !="target"]
        options2 = st.multiselect(
        'Select features to use in the naive bayes',
        allowed_options2,
        allowed_options2)
        test_ratio_naive_bayes = st.slider('Select ratio of the naive bayes test samples', 0.1, 0.5, 0.2, 0.05)
        st.button("Reset Naive Bayes") 
        run_naive_bayes_state = st.button('Run Naive Bayes')
        if run_naive_bayes_state:
            st.write('You selected:', options2)
            X2 = df[options2].values
            y2 = df.iloc[:,0].values.reshape(-1, 1)
            X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=test_ratio_naive_bayes, random_state = 42) 
            encX2.fit(X_train)
            X_enc2 = encX2.fit_transform(X_train)
            ency2.fit(y_train)
            y_enc2 = ency2.fit_transform(y_train)
            X_test_enc2 = encX2.fit_transform(X_test)
            y_test_enc2 = ency2.fit_transform(y_test)
            model = CategoricalNB()
            model.fit(X_enc2, y_enc2)
            y_train_pred_naive_bayes = model.predict(X_enc2)
            naive_bayes_accuracy = 100 - np.sum(np.abs(y_train_pred_naive_bayes-y_enc2.flatten()))/X_enc2.shape[0]*100
            # st.markdown(f"Accuracy = {naive_bayes_accuracy}")
            y_test_pred_naive_bayes = model.predict(X_test_enc2)
            # st.markdown(f"Testing accuracy = {accuracy_score(y_test_enc2, y_test_pred_naive_bayes)*100}")
            results = pd.DataFrame([[naive_bayes_accuracy, accuracy_score(y_test_enc2, y_test_pred_naive_bayes)*100]], columns = ["Training Accuracy","Testing Accuracy"])
            colr1, colr2, colr3 = st.columns([0.3, 0.5, 0.2])
            with colr2:
                st.dataframe(results)
            fig_conf__naive_bayes, conf_axes_naive_bayes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            ConfusionMatrixDisplay.from_estimator(model, X_test_enc2,y_test_enc2, ax = conf_axes_naive_bayes)
            colrrr1, colrrr2, colrrr3 = st.columns([0.3, 0.5, 0.2])
            with colrrr2:
                st.subheader("Confusion Matrix")
            st.pyplot(fig_conf__naive_bayes) 
        else:
            st.write('Please run the naive bayes')
        

with tab_Conclusion:


    st.markdown('#')
    st.markdown('#')
    st.markdown('In conclusion, the study demonstrates that it is possible to effectively categorize mushrooms as either edible or poisonous by utilizing a carefully selected set of six features. One such set comprises the following characteristics: odor, spore-print-color, stalk-color-above-ring, stalk-surface-below-ring, population, and ring-type. However, it is important to note that machine learning algorithms, despite achieving a high level of accuracy exceeding 94%, are not infallible. When all features are considered, only the decision tree algorithm achieves flawless performance, even when subjected to a testing subset with a 50% size. In summary, both the decision tree and support vector machine algorithms exhibit nearly identical accuracy levels, surpassing the performance of the naive Bayes algorithm.')
    image2 = Image.open('classification.png')
    st.image(image2, caption='Classification of Mushrooms Tree.')
    st.markdown('#')
    st.markdown('#')
    exp = hip.Experiment.from_dataframe(df[['target', 'ring-type', 'population',\
                     'stalk-color-above-ring', 'stalk-surface-below-ring',\
                           'spore-print-color','odor' ]])
    htmlcomp = exp.to_html()
    st.components.v1.html(htmlcomp,width=700, height=600, scrolling=True)
    
    
    st.subheader("Accuracy tables with the selected features")
    accuracy_chosen = [[99.43, 99.32], [99.43, 99.38], [99.54, 99.38], [98.6, 98.8], [97.4, 97.3], [96.1, 95.5]]
    index_chosen = ["Decision Tree %20 for testing","Decision Tree %50 for testing","Support Vector Machine %20 for testing","Support Vector Machine %50 for testing","Naive Bayes %20 for testing","Naive Bayes %50 for testing"]
    columns = ["Train Accuracy", "Test Accuracy"]
    colac1, colac2, colac3 = st.columns([0.1, 0.7, 0.2])
    with colac2:
        st.dataframe(pd.DataFrame(accuracy_chosen, index = index_chosen, columns = columns))
    
    
    st.subheader("Accuracy tables with all the features selected")
    accuracy_full = [[100, 100], [100, 100], [99.83, 99.57], [99.56, 99.29], [95.5, 95.1], [95, 94.6]]
    index_full = ["Decision Tree %20 for testing","Decision Tree %50 for testing","Support Vector Machine %20 for testing","Support Vector Machine %50 for testing","Naive Bayes %20 for testing","Naive Bayes %50 for testing"]
    columns = ["Train Accuracy", "Test Accuracy"]
    colacc1, colacc2, colacc3 = st.columns([0.1, 0.7, 0.2])
    with colacc2:
        st.dataframe(pd.DataFrame(accuracy_full, index = index_full, columns = columns))
    
    
    st.subheader("FP tables with the selected features")
    FP_chosen = [[11, 0], [25, 0], [2, 8], [38, 9], [44, 0], [184, 0]]
    index_chosen = ["Decision Tree %20 for testing","Decision Tree %50 for testing","Support Vector Machine %20 for testing","Support Vector Machine %50 for testing","Naive Bayes %20 for testing","Naive Bayes %50 for testing"]
    columns = ["FN", "FP"]
    colaccc1, colaccc2, colaccc3 = st.columns([0.1, 0.7, 0.2])
    with colaccc2:
        st.dataframe(pd.DataFrame(FP_chosen, index = index_chosen, columns = columns))
    
    
    st.subheader("FP tables with all the features selected")
    FP_full = [[0, 0], [0, 0], [6, 1], [20, 9], [74, 6], [212, 7]]
    index_full = ["Decision Tree %20 for testing","Decision Tree %50 for testing","Support Vector Machine %20 for testing","Support Vector Machine %50 for testing","Naive Bayes %20 for testing","Naive Bayes %50 for testing"]
    columns = ["FN", "FP"]
    colacccc1, colacccc2, colacccc3 = st.columns([0.1, 0.7, 0.2])
    with colacccc2:
        st.dataframe(pd.DataFrame(FP_full, index = index_full, columns = columns))
    
    
    

with tab_bio:
    
    st.markdown('#')
    st.markdown('#')
    st.markdown('I am currently enrolled as a doctoral philosophy student specializing in mathematics. My research focus primarily centers around the development of topological data analysis methodologies that have practical applications within the domain of molecular biology. I maintain a strong interest in diverse mathematical subjects, with a particular passion for the field of algebra.')
    st.markdown('#')
    st.markdown('#')
    
