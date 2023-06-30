#!/usr/bin/env python
# Created by "Thieu" at 07:35, 30/06/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mafese import get_dataset
from mafese.wrapper.mha import MultiMhaSelector

data = get_dataset("Arrhythmia")
data.split_train_test(test_size=0.2)

list_paras = [{"epoch": 10, "pop_size": 30}, ]*4

feat_selector = MultiMhaSelector(problem="classification", estimator="knn",
                            list_optimizers=("OriginalWOA", "OriginalGWO", "OriginalTLO", "OriginalGSKA"), list_optimizer_paras=list_paras,
                            transfer_func="vstf_01", obj_name="AS")

feat_selector.fit(data.X_train, data.y_train, n_trials=5, n_jobs=5, verbose=False)
feat_selector.export_boxplot_figures()
feat_selector.export_convergence_figures()

# X_selected = feat_selector.transform(data.X_test)
# print(f"Original Dataset: {data.X_train.shape}")
# print(f"Selected dataset: {X_selected.shape}")



# import the required library
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# print(plt.style.available)
#
# plt.style.use("seaborn")
#
# # load the dataset
# df = pd.read_csv("history/best_fitness.csv")
# df.boxplot(by='model', column=['best_fitness'], grid=False)
# plt.ylabel("Global best fitness value")
# plt.xlabel("Model")
# plt.title("Amazing")
# plt.suptitle('')
# plt.tight_layout()
# plt.show()





#
# import plotly.express as px
# import plotly.io as pio
# pio.kaleido.scope.mathjax = None
#
# import pandas as pd
#
# df = pd.read_csv("history/best_fitness.csv")
# n_trial = len(df["trial"].unique())
# fig = px.box(df, x="model", y="best_fitness", color="model",
#              labels={
#                  "model": "Model",
#                  "best_fitness": "Global best fitness value"})
# fig.update_traces(boxmean="sd") # or "inclusive", or "linear" by default
# fig.update_layout(
#     margin=dict(l=20, r=20, t=30, b=20),
#     title={
#                  'text': f"Boxplot after {n_trial} trials of comparison models",
#                  # 'y': 0.9,
#                  'x': 0.5,
#                  'xanchor': 'center',
#                  'yanchor': 'top'},
#     showlegend=False
# )
# # fig.show(renderer="svg")
# fig.write_image("fig1.png")



# import pandas as pd
# import plotly.express as px
#
# # Step 2: Load the CSV file into a Pandas DataFrame and reset the index
# df = pd.read_csv("history/convergence-trial1.csv")
# df = df.reset_index()
#
# # Step 3: Melt the DataFrame to convert it from wide to long format
# df_long = pd.melt(df, id_vars='index', var_name='Column', value_name='Value')
#
# # Step 4: Define the line chart using Plotly Express
# fig = px.line(df_long, x='index', y='Value', color='Column',
#               labels={'index': 'Epoch', 'Value': 'Fitness value', 'Column': 'Model'})
# fig.update_layout(
#     margin=dict(l=20, r=20, t=30, b=20),
#     title={
#         'text': 'Line Chart for Dataset',
#         # 'y': 0.9,
#         'x': 0.5,
#         'xanchor': 'center',
#         'yanchor': 'top'},
#     showlegend=True
# )
# # Step 5: Display the plot using Plotly Express
# # fig.show()
# fig.write_image("fig0.png")
