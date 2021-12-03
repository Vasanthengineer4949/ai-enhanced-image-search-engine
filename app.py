import streamlit as st
import pandas as pd, numpy as np
from html import escape
import os
from transformers import CLIPProcessor, CLIPTextModel, CLIPModel
@st.cache(show_spinner=False,
          hash_funcs={CLIPModel: lambda _: None,
                      CLIPTextModel: lambda _: None,
                      CLIPProcessor: lambda _: None,
                      dict: lambda _: None})
def load():
  model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
  processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
  df = {0: pd.read_csv('./resources/dataset/data.csv'), 1: pd.read_csv('./resources/dataset/data2.csv')}
  embeddings = {0: np.load('./resources/embeddings/embeddings.npy'), 1: np.load('./resources/embeddings/embeddings2.npy')}
  for k in [0, 1]:
    embeddings[k] = np.divide(embeddings[k], np.sqrt(np.sum(embeddings[k]**2, axis=1, keepdims=True)))
  return model, processor, df, embeddings
model, processor, df, embeddings = load()
source = {0: '\nSource: Unsplash', 1: '\nSource: The Movie Database (TMDB)'}
def get_html(url_list, height=200):
    html = "<div style='margin-top: 20px; max-width: 1200px; display: flex; flex-wrap: wrap; justify-content: space-evenly'>"
    for url, title, link in url_list:
        html2 = f"<img title='{escape(title)}' style='height: {height}px; margin: 5px' src='{escape(url)}'>"
        if len(link) > 0:
            html2 = f"<a href='{escape(link)}' target='_blank'>" + html2 + "</a>"
        html = html + html2
    html += "</div>"
    return html
def compute_text_embeddings(list_of_strings):
    inputs = processor(text=list_of_strings, return_tensors="pt", padding=True)
    return model.get_text_features(**inputs)
st.cache(show_spinner=False)
def image_search(query, corpus, n_results=24):
    text_embeddings = compute_text_embeddings([query]).detach().numpy()
    k = 0 if corpus == 'Unsplash' else 1
    results = np.argsort((embeddings[k]@text_embeddings.T)[:, 0])[-1:-n_results-1:-1]
    return [(df[k].iloc[i]['path'],
             df[k].iloc[i]['tooltip'] + source[k],
             df[k].iloc[i]['link']) for j in range(1) for i in results]
description = '''
# CLIP image search
**Enter the kind of image you need and then press Enter after clicking which you need Unsplash image or movie image**
*Built with OpenAI's [CLIP](https://openai.com/blog/clip/) model, 🤗 Hugging Face's [transformers library](https://huggingface.co/transformers/), 
*Inspired from vivien
***Made with ❤️ by [Vasanth](https://www.linkedin.com/in/vasanthengineer4949/)***
'''
def main():
  st.markdown('''
              <style>
              .block-container{
                max-width: 1500px;
              }
              div.row-widget.stRadio > div{
                flex-direction:row;
                display: flex;
                justify-content: center;
              }
              div.row-widget.stRadio > div > label{
                margin-left: 10px;
                margin-right: 1-px;
              }
              section.main>div:first-child {
                padding-top: 0px;
              }
              section:not(.main)>div:first-child {
                padding-top: 50px;
              }
              div.reportview-container > section:first-child{
                max-width: 380px;
              }
              #MainMenu {
                visibility: hidden;
              }
              footer {
                  visibility: visible;
                  text-align: center;
                  }
              footer:after{
                content: "Made with ❤️ by Vasanth, Inspired from vivien";
                display: block;
                position: relative;
                justify-content: center;
                color: tomato;
                text-align: center;
                }
              </style>''',
              unsafe_allow_html=True)
  st.sidebar.markdown(description)
  _, c, _ = st.columns((1, 3, 1))
  query = c.text_input('', value='a beach where there is a beautiful sunset')
  corpus = st.radio('', ["Unsplash","Movies"])
  if len(query) > 0:
    results = image_search(query, corpus)
    st.markdown(get_html(results), unsafe_allow_html=True)
if __name__ == '__main__':
  main()