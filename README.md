# 🧪 Behavioural Testing on a Sentiment Clasiffier 🔬

Check how robust a sentiment classifier is to random typos in our dataset using Github Actions.

## 🗃️ Table of contents

- [🗺️ A bit of context about this project](#🗺️-a-bit-of-context-about-this-project)
- [📝 Description](#📝-description)
- [🛠️ Set up](#🛠️-set-up)
- [🌱 How to use Github Actions to test this and your own project](#🌱-how-to-use-github-actions-to-test-this-and-your-own-project)
- [👥 Aknowledgements](#👥-aknowledgements)

## 🗺️ A bit of context about this project

[Back to Top](#🧪-behavioural-testing-on-a-sentiment-clasiffier-🔬)

This project represents an aid for the talk **"Testing Schrödinger's Box, AKA ML Systems"** at [CommitConf 2024](https://koliseo.com/commit/2024/agenda/0) and [Codemotion 2024](https://conferences.codemotion.com/madrid2024/?utm_source=google_ads&utm_medium=paid_search&utm_campaign=CONFC_ESP_CODEMOTION_2024_MADRID&utm_content=esp&source=adv_google_search&gad_source=1).

<p align="center">
  <img width="600" src="./data/project_icon.png">
</p>

**Abstract of the talk:**

*Just like quantum mechanics and Schrödinger's cat experiment, in AI we have our own mysteries, which, curiously, are also related to boxes.We have come to create systems of such complexity that we call them "black box models" because we are unable to understand what goes on inside them. We only know what goes in and what comes out.*

*In this talk, we will talk about how can we test these black boxes to shed some light on what goes on inside them, or at least to ensure that they behave in a predictable way. Which is not trivial at all. We will also discuss a hands on example on how perform automatic tests on a sentiment anañlysis project.*

## 📝 Description

[Back to Top](#🧪-behavioural-testing-on-a-sentiment-clasiffier-🔬)

In this project we present the code to predict an individual's belief about climate change based on their Twitter activity.

- Information about the dataset and data processing performed: ```data/README.md``` and ```docs/processed/data_processing.ipynb```
- Benchmarking of the model: ```docs/processed/sentiment_analysis_guide.ipynb```
- Perturbation test: ```run_perturbation_test.py``` will asses the robustness of our classifier to typos in our dataset.

## 🛠️ Set up

[Back to Top](#🧪-behavioural-testing-on-a-sentiment-clasiffier-🔬)

1. This project requires python>=3.7=<3.11. Check your python version by running:

```bash
python --version
```

2. Clone the repository:

```bash
git clone https://github.com/LoboaTeresa/Behavioural-Testing-on-a-Sentiment-Clasiffier.git
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. You are ready to go! Dont forget to check the notebooks in the ```docs``` folder to understand the data processing and the benchmarking of the model.

## 🌱 How to use Github Actions to test this and your own project

[Back to Top](#🧪-behavioural-testing-on-a-sentiment-clasiffier-🔬)

Learning journey on Github Actions: click [here](https://resources.github.com/learn/pathways/automation/?utm_campaign=copilot-banner&utm_medium=Resources&utm_source=learning-pathways)

1. Create your own Github repository or fork this one. Github actions are integrated into your project as soon you create the Github repository.

2. Click on Actions in the top bar of your repository to check pre-built workflows. As you can see Github actions offers a very easy integratios with different tools, something indespensable for CI/CD tool.

3. Your automated workflows must be defined in ```.github/workflows/test.yaml``` file. You can change the name of the file, but it must be inside the ```.github/workflows``` folder. Check out mine.

4. You can modify the name of the job, the name of the workflow, the name of the python version, the name of the test, etc. You can also add more jobs to the workflow.

5. Once you have created the yaml file, you can push it to your repository. This will trigger the workflow and you will be able to see the results in the Actions tab of your repository.

## 👥 Aknowledgements

[Back to Top](#🧪-behavioural-testing-on-a-sentiment-clasiffier-🔬)

I would like to thank the organizers of CommitConf and Codemotion for giving me the opportunity to share my knowledge with the community. I would also like to thank the community for their support and feedback.

The code in this repository is based on the work of [Max Stocker](https://github.com/m-stock/climate_tweets_nlp/tree/main). Go give him a star in his repository and some claps for his [Medium Article](https://medium.com/@max.h.stocker/sentiment-analysis-of-climate-tweets-2ea31724ad87).

<img src="https://blog.commit-conf.com/content/images/2018/04/commit-white-1.png" width="200" alt="commitconf" title="CommitConf Logo">  

<img src="https://extra.codemotion.com/app/uploads/2019/02/Codemotion_2018_logo_orange_white_1500x300_RGB-2.png" width="300" alt="codemotion" title="Codemotion Logo">
