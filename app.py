import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader,YoutubeLoader
#from langchain_yt_dlp.youtube_loader import YoutubeLoaderDL
#from langchain.document_loaders import 

# loader= YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=HJXWpqpcHik")
# #, languages=["en","en-US"])
# transcript = loader.load()
# print(transcript)

## streamlit app
st.set_page_config(page_title="Langchain: Summarize Text from YT or Website")
st.title("Langchain: Summarize Text from YT or Website")
st.subheader("Summarize URL")

#Groq api and url to be summarized
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url= st.text_input("URL",label_visibility="collapsed")

llm = ChatGroq(api_key=st.secrets["GROQ_API_KEY"], model="Gemma2-9b-It")

promt_template="""
Provide a summary of the following content in 300 words:
Content"{text}
"""
prompt=PromptTemplate(template=promt_template, input_variables=["text"])


if st.button("Summarize the Content from YT or Website"):
    ## Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url. It can may be a YT video or website Url")

    else:
        try:
            with st.spinner("Waiting..."):
                ## loading the website or yt video data
                if "youtube.com" in generic_url:
                    loader= YoutubeLoader.from_youtube_url(generic_url)
                    #YoutubeLoaderDL.from_youtube_url(generic_url,add_video_info=True)

                else:
                    loader= UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                  header={"User-Agent":"Mozilla/5.0"})
                data=loader.load()
                #st.write(data)

                ## Chain for summarization
                chain=load_summarize_chain(llm,chain_type="stuff", prompt=prompt)
                output_summary=chain.run(data)
                #st.write(data[0].metadata)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception:{e}")

                    
    