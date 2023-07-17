import streamlit as st
#from langchain.llms import OpenAI
import os
import openai
import pandas as pd
import networkx as nx
import matplotlib.pyplot  as plt
import holoviews as hv
from holoviews import opts, dim
hv.extension('bokeh')  # Explicitly load the Bokeh plotting extension

from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

openai.api_key = "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"

st.title('Build Partnership Networks using OpenAI''s GPT LLM')

st.write('Hello world!')

openai_api_key = st.sidebar.text_input('Enter your OpenAI API KEy here : ')

def generate_response(input_text):
  response = openai.Completion.create(
    model="text-davinci-003",
    prompt=input_text,
    temperature=0,
    max_tokens=1000,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )
  
  #st.info(type(response.choices[0].text))

  st.info(response.choices[0].text)

  resp_table_text = response.choices[0].text

  # Split the string by newlines to get individual rows
  rows = resp_table_text.strip().split('\n')
  #for row in rows :
    #st.info("row") 
    #st.info(row)
  # Extract column names from the first row
  column_names = rows[0].split(',')
  #st.info("column_names") 
  #st.info(column_names)
  # Create an empty dictionary to store the data
  data = {}

  # Iterate over the remaining rows
  for row in rows[1:]:
      # Split each row by commas to get column values
      #st.info("values") 
      values = row.split(',')
      #st.info(values)
      # Iterate over the column names and values simultaneously
      for column, value in zip(column_names, values):
          # Check if the column already exists in the dictionary
          if column in data:
              if value is not None:
                data[column].append(value.strip())
              #else:
                #data[column].append('N/A')
          else:
              # If the column doesn't exist, create a new list with the value
              if value is not None:
                data[column] = [value.strip()]
              #else:
                #data[column].append('N/A')
          #st.info("data[column]")
          #st.info(data[column])    
      #st.info("data complete dictionary outside loop") 
      #st.info(data)
  # Create a DataFrame from the dictionary
  df = pd.DataFrame(data)

  #df = pd.DataFrame(resp_table_text, columns=["Company1" , "Company2" , "Year" , "PartnershipType"])
  #df = [n.split('|') for n in resp_table_text]
  #df = pd.DataFrame(df[1:], columns=df[0])
  
  #st.info("Dataframe")
  #st.info(df)
  AgGrid(df)

  #st.info("response")
  #st.info(response)


  #st.info("response chices")
  #st.info(response.choices)

  #st.info("response first object in choices")
  #st.info(response.choices[0])

  #st.info("response first object in choices - text ")
  #st.info(response.choices[0].text)


  #st.info("data frame's column")
  #st.info(df.columns)
  columnlist = df.columns
  st.info(df.columns)
  st.info(columnlist)

  st.info(columnlist[0])
  st.info(columnlist[1])
  st.info(columnlist[2])

  #st.info(df[columnlist[0]])
  
  # Create the DataFrame
  #df_edges = pd.DataFrame(edge_data)
  df_edges = df.iloc[: , :3]
  st.info(df_edges)


  st.info(df_edges.iloc[:0])
  

  #st.info(df_edges)

  # Create an empty graph
  G = nx.from_pandas_edgelist(df_edges, source=columnlist[0], target=columnlist[1] , edge_attr = columnlist[2] )

  # Customize graph layout
  pos = nx.spring_layout(G)


  # Draw the network graph
  plt.figure(figsize=(8, 6))
  nx.draw(G, pos, with_labels=True, node_color='green', edge_color='gray')



  # Display the graph using Streamlit
  st.pyplot(plt)

  # Create a Holoviews Graph object
  hv_graph = hv.Graph.from_networkx(G, nx.spring_layout)

  # Customize the appearance of the graph
  hv_graph.opts(
    node_color='lightblue',
    edge_color='gray',
    node_size=20,
    edge_line_width=1,
    width=500,
    height=400,
  )

  # Display the graph using Streamlit - - This needs to be fixed 
  st.write(hv_graph)

  # Group the DataFrame by 'Year' and count the number of rows in each year
  df_count = df.groupby([columnlist[2]]).size().reset_index(name='Count')

  AgGrid(df_count)
 
  #st.write(weekly_data)
  #Bar Chart
  st.bar_chart( df_count['Count'])
  # Set x-axis labels to be the years
  x_ticks = df_count[columnlist[2]]
  st.pyplot().set_xticklabels(x_ticks)
  st.title('Number of Partnerships by Year')
  st.xlabel('Year')
  st.ylabel('Number of Partnerships')

  # RADAR CHART CODE !!!!!!!!!!!
  import numpy as np
  #import matplotlib.pyplot as plt
  from math import pi


  # obtain df information
  categories = list(df)[1:]
  values = df.mean().values.flatten().tolist()
  values += values[:1] # repeat the first value to close the circular graph
  angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
  angles += angles[:1]

  # define plot
  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8),
                          subplot_kw=dict(polar=True))
  plt.xticks(angles[:-1], categories, color='grey', size=12)
  plt.yticks(np.arange(0.5, 2, 0.5), ['0.5', '1.0', '1.5'],
            color='grey', size=12)
  plt.ylim(0, 2)
  ax.set_rlabel_position(30)

  # draw radar-chart:
  for i in range(len(df)):
      val_c1 = df.loc[i].drop('Name').values.flatten().tolist()
      val_c1 += val_c1[:1]
      ax.plot(angles, val_c1, linewidth=1, linestyle='solid',
              label=df.loc[i]["Name"])
      ax.fill(angles, val_c1, alpha=0.4)

  # add legent and show plot
  plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
  plt.show()



with st.form('my_form'):
  text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
  submitted = st.form_submit_button('Submit')
  if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
  if submitted and openai_api_key.startswith('sk-'):
    generate_response(text)













