

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


# df = pd.read_csv('/Users/faisalsuwayyid/Desktop/Projects/CMSE_Midterm/The project/7_Mushroom/Mushroom.csv')
# df = df.iloc[:,1:]

st.set_page_config(page_title = 'Classification of Mushrooms to Edible and Poisonous',
                   page_icon = 'bar_chart',
                   layout='wide')


col1, col2, col3 = st.columns([0.2, 0.6, 0.2])
with col2:
    st.title('Classification of Mushrooms to Edible and Poisonous')
    tab1, tab2, tab3, tab4 = st.tabs(["Homepage", "Features Describtion and Study Validation", "Analysis", "Conclusion Remarks"]) 

    with tab1:
        st.header("The project aim:")
        st.markdown("In this project we classify certain family of mushrooms called the Agaricus and Lepiota Family \
                    into edible and poisonous based on their atanomy and other featurs. \
                    We have dataset containing 8124 samples characterized by 22 features describing them.")
        st.header("Who is this app for:")
        st.markdown("This app is generally for the public audience who are interested in learning how to \
                    distinguish between edible and poisonous mushrooms. However, it is useful for:")
        st.markdown(" * People who are interested in hunting mushrooms.")
        st.markdown(" * People who like to travel into the wild and think they might encounter mushrooms.")

        st.header("Significance:")
        st.markdown("The data shows there is a way to classify the mushrooms in the data with absolute certainty to\
                    edible and poisonous since there is no edible and poisonous mushrooms with the same features.")
        
        st.header("References:")
        st.markdown("The dataset is obtained from UCI repository, and is called Mushroom. Here is a link to the \
                    dataset: https://archive.ics.uci.edu/dataset/73/mushroom")
    
    with tab2:

        st.header("Features:")
        image = Image.open('AdobeStock_321292924_Preview copy.jpg')
        st.image(image, caption='Mushroom Anatomy')
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
        st.markdown('In the below plot, we notice that we have poisonous \
                    mushrooms almost as much as edible ones. \
                    This suggests that, under the assumption of a uniform distribution, \
                    it is very likely to encounter a poisonous mushrooms.')
        st.markdown('#')
        st.markdown('#')
        ####### Data Validation #########
        col1, col2, col3 = st.columns(3)
        with col2:
            st.markdown('# Size of data')

        option_data_validation = st.selectbox(
        "Select a feature to view its histogram",
        tuple([None, 'percent']),
        index=0,
        placeholder="Select contact method...",
        )
        fig = px.histogram(df, x="target", text_auto=True,
                            histnorm = option_data_validation, histfunc = 'count')
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        ###############################

        st.markdown('We investigate the histogram of each \
                    feature to understand if there is any deterministic \
                    feature that characterizes mushrooms to edible and poisonous \
                    and how mushrooms are distributed over classes of the feature selected. \
                    Generally, we look for features where edible and poisonous have less common \
                    classes. Additionally, if possible, we also look for the cases where \
                    poisonous or edible mushrooms are very few in the common classes.')
        ############ Histogram count ###############
        
        st.markdown('# The percentage of classes in features')
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
        st.markdown('The below plot shows the percentage of both \
                    poisonous and edible mushrooms in the same class\
                    in the selected feature.')
        # relative_histplot(df, option)

        fig = px.histogram(df, x=option_data_histogram,
                            text_auto=True, color = 'target', barnorm = 'percent')
        st.plotly_chart(fig, use_container_width=True)

        ##############################
        st.markdown('#')
        st.markdown('#')
        st.markdown('Going over the features, we observe that \
                    odor, spore-print-color, veil-color and population satisfy our criteria. \
                    odor alone seems like a good start as it shows most of the poisonous mushrooms \
                    have odor, and very few are with no odor.')
        st.markdown('#')
        st.markdown('#')



        ##############################
        col1, col2, col3 = st.columns([0.15, 0.84, 0.01])
        with col2:
            st.markdown('# Pair of features study')
        st.markdown('Here, we investigate the distribution and probability of selected pair of features. \
                    The below table shows the probablities of the pair of classes of the selected two features.\
                    The left table shows the probability of edible mushrooms over the pair of classes of\
                    the pair of selected features. The right table shows the ones for the poisonous mushrooms.\
                    Zero percentage corresponding to a pair of classes in the left table \
                    implies that all mushrooms with such pair of classes are \
                    poisonous, if any, and vice versa.')
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
        st.markdown('After experimenting, we conclude that (odor, spore-print-color) \
                    is good as there is only one common pair between edible and poisonous \
                    with nonzero probability. Generally, using pairs of features reduces the probability \
                    significantly.')





        # ######### Sunburst1 plot ############
        st.markdown('#')
        st.markdown('#')
        st.markdown('We use a sunburst plot to display a 4 dimensional plot to help in understanding \
                    how three features can classify mushrooms. We are mainly interested in \
                    odor, spore-print-color, and veil color or population.')
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



#############################################



with tab4:
    st.markdown('#')
    st.markdown('#')
    st.markdown('In conclusion, we find that to classify mushrooms \
                to edible and poisonous, it is sufficient to use seven features \
                selected in a good way. One such group of seven features is \
                (odor, spore-print-color, veil-color, stalk-color-above-ring, \
                stalk-surface-below-ring, ring-type and population).')
    st.markdown('#')
    st.markdown('#')
    exp = hip.Experiment.from_dataframe(df[['target', 'population', 'ring-type',\
                 'stalk-color-above-ring', 'stalk-surface-below-ring',\
                      'veil-color', 'spore-print-color','odor' ]])
    htmlcomp = exp.to_html()
    st.components.v1.html(htmlcomp,width=700, height=600, scrolling=True)






