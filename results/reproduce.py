import os.path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import warnings
from pathlib import Path

pd.set_option("plotting.backend", "plotly")  # -- set some plotly default stuff
warnings.filterwarnings("ignore")  # -- supress pandas split warnings

# -- constants defining plot estetics
COLOR_SIMSIAM = '#00CC96'
COLOR_SIMSIAM_SUPERVISED = '#7fe5ca'
COLOR_SIMTRIPLET = '#EF553B'
COLOR_SIMTRIPLET_SUPERVISED = '#FEAB9A'
COLOR_SIMCLR = '#FFA15A'
COLOR_SIMCLR_SUPERVISED = '#ffd0ac'
COLOR_PAWS = '#636EFA'
COLOR_PAWS_SUPERVISED = '#b1b6fc'

COLOR_MAPS = {
    'method': {
        'Supervised': '#AB63FA',
        # PAWS
        'paws': COLOR_PAWS,
        'PAWS': COLOR_PAWS,
        # SimTriplet
        'simtriplet': COLOR_SIMTRIPLET,
        'SimTriplet': COLOR_SIMTRIPLET,
        # SimSiam
        'simsiam': COLOR_SIMSIAM,
        'SimSiam': COLOR_SIMSIAM,
        # SimCLR
        'simclr': COLOR_SIMCLR,
        'SimCLR': COLOR_SIMCLR,
    }
}

VAL_TO_LABEL = {
    "True": "freezed",
    "900": "finetuned",
    "simtriplet": "SimTriplet",
    "simsiam": "SimSiam",
    "simclr": "SimCLR",
    "paws": "PAWs",
    "PCamPCam": "P -> P",
    "KatherPCam": "N -> P",
    "PCamKather": "P -> N",
    "KatherKather": "N -> N",
    "KatherLizard": "N -> L",
    "PCamLizard": "P -> L",
    "Falseresnet18one_layer_mlp": "R18,1L,T",
    "Falseresnet18three_layer_mlp": "R18,3L,T",
    "Falseresnet50one_layer_mlp": "R50,1L,T",
    "Falseresnet50three_layer_mlp": "R50,3L,T",
    "Falsewide_resnet28w2one_layer_mlp": "W28,1L,T",
    "Falsewide_resnet28w2three_layer_mlp": "W28,3L,T",
    "Trueresnet18one_layer_mlp": "R18,1L,F",
    "Trueresnet18three_layer_mlp": "R18,3L,F",
    "Trueresnet50one_layer_mlp": "R50,1L,F",
    "Trueresnet50three_layer_mlp": "R50,3L,F",
    "Truewide_resnet28w2one_layer_mlp": "W28,1L,F",
    "Truewide_resnet28w2three_layer_mlp": "W28,3L,F",
    "test_acc": "Accuracy",
    "test_f1": "f1-score",
}

COL_TO_LABEL = {
    'test_ece': 'Expected Calibration Error (ECE)',
    'enc_ds': 'Encoder',
    'test_acc': 'Maximum Accuracy',
    'split_size': 'Split Size',
    'kather_h5_224_norm_split_90_10': 'Kather',
    'patchcamelyon': 'PCam',
    'lizard': 'Lizard',
    'benefit_test_acc': 'Benefit: Accuracy',
    'benefit_test_f1': 'Benefit: f1-score',
    'theta': 'Encoder Training Data -> Downstream Task',
    'theta2': 'Architecture Settings',
    "simtriplet": "SimTriplet",
    "simsiam": "SimSiam",
    "simclr": "SimCLR",
    "paws": "PAWs",
}


def calculate_benefits(df: pd.DataFrame) -> pd.DataFrame:
    """ Calculate the benefites over all training metrics.
    """
    benefit_metrics = ['auroc', 'test_acc', 'test_auroc', 'test_f1', 'v_acc', 'v_f1']
    for metric in benefit_metrics:
        df['benefit_' + metric] = np.nan

    for encoder in df.encoder.unique():
        for freeze in df.freeze_encoder.unique():
            for down_dataset in df.down_ds.unique():
                for enc_dataset in df.enc_ds.unique():
                    for split in df.split_size.unique():
                        for pred_head in df.pred_head_structure.unique():
                            for method in df.method_plain.unique():
                                # -- query supervised points for one plot
                                idx_method = (df.encoder == encoder) & (df.freeze_encoder == freeze) & (df.down_ds == down_dataset) & (df.enc_ds == enc_dataset) & (df.split_size == split) & (df.pred_head_structure == pred_head) & (df.method == method)
                                idx_supervised = (df.encoder == encoder) & (df.freeze_encoder == freeze) & (df.down_ds == down_dataset) & (df.enc_ds == enc_dataset) & (df.split_size == split) & (df.pred_head_structure == pred_head) & (df.method == method + '-supervised')

                                # -- when calculating benefits of frozen runs, create a second benefit calculated against the unfrozen supervised run
                                if freeze == 'True':
                                    idx_supervised_unfreezed = (df.encoder == encoder) & (df.freeze_encoder != freeze) & (df.down_ds == down_dataset) & (df.enc_ds == enc_dataset) & (df.split_size == split) & (df.pred_head_structure == pred_head) & (df.method == method + '-supervised')

                                for metric in benefit_metrics:
                                    a = df.loc[idx_method, metric]
                                    b = df.loc[idx_supervised, metric]
                                    if not a.empty and not b.empty:
                                        df.loc[idx_method, 'benefit_' + metric] = a.iat[0] - b.iat[0]

                                    if freeze == 'True':
                                        c = df.loc[idx_supervised_unfreezed, metric]
                                        if not a.empty and not c.empty:
                                            df.loc[idx_method, 'benefit_against_unfreezed_run_' + metric] = a.iat[0] - c.iat[0]
    return df


def filter_df(df: pd.DataFrame, query: str, sort: dict = None, old_paws_config_bug_fix: bool = False) -> pd.DataFrame:
    """Appling a pandas query or sorting to a dataframe and possibly fix
       old PAWS configs with unused parameters.
    """
    # In old paws configs the values was set to false but not used in the code
    if old_paws_config_bug_fix:
        df.loc[df['method'] == 'paws', 'keep_enc_fc'] = True
        df.iloc[df['method'] == 'paws-supervised', 'keep_enc_fc'] = True

    df.query(query, inplace=True)
    if sort is not None:
        df.sort_values(**sort, inplace=True)

    return df


def create_split_reduce_plots(csv_path: str, result_path: str) -> None:
    """ Reproduce several figures from the main text and appendix, including
        Fig.3, Fig.4, Fig.6, A.1(a)-(f), A.2
    """
    # -- define metrics to plot
    metrics = ['test_acc', 'test_f1']

    # -- calculate benefits and remove supervised runs
    df = pd.read_csv(csv_path)
    df = calculate_benefits(df)
    df = df[~df['method'].str.contains('supervised')]

    for filter in ['all', 'uf_vs_uf', 'f_vs_f']:
        RESULTS_PATH_FOLDER = Path(result_path, filter)
        if not os.path.exists(RESULTS_PATH_FOLDER):
            os.makedirs(RESULTS_PATH_FOLDER)

        if filter == 'all':
            df_plot = df
        elif filter == 'uf_vs_uf':
            df_plot = df[df.freeze_encoder != 'True']
        elif filter == 'f_vs_f':
            df_plot = df[df.freeze_encoder != 'False']

        df_plot['theta'] = df_plot['enc_ds'] + df_plot['down_ds'] + '_' + df_plot['method']
        df_plot['theta_sorting'] = df_plot['method'] + df_plot['encoder'] + "-" + df_plot['split_size'].astype(str)
        df_plot.loc[df_plot['freeze_encoder'] != 'True', "freeze_encoder"] = 'False'
        df_plot['theta2'] = df_plot['freeze_encoder'] + df_plot['encoder'] + df_plot['pred_head_structure']
        df_plot['theta2_sorting'] = df_plot['method'] + df_plot['freeze_encoder'] + df_plot['encoder'] + df_plot['pred_head_structure']

        for metric in metrics:
            # -- global layout modifications
            layout_update_legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title_text='')
            layout_update_custom_global = dict(
                showlegend=True,
                autosize=False,
                width=800,
                height=800,
                yaxis_title=f"Benefit:  {VAL_TO_LABEL[metric]}",
                font=dict(size=16),
                boxmode='group',
                legend=layout_update_legend,
            )

            # -- sorting
            df_plot.sort_values(by=['theta_sorting'], inplace=True)
            column_to_iterate_over = 'method_label'
            fig = go.Figure()
            for value in df_plot[column_to_iterate_over].unique():
                _stb_df = df_plot[df_plot[column_to_iterate_over] == value]
                _stb_df.sort_values(by=['down_ds', 'split_size'], inplace=True)
                fig.add_trace(
                    go.Box(
                        x=[list(_stb_df['down_ds']), list(_stb_df['split_size'])],
                        y=list(_stb_df[f"benefit_{metric}"]),
                        line=dict(color=COLOR_MAPS['method'][value.lower()]),
                        name=value,
                    )
                )
            fig.update_layout(layout_update_custom_global)
            fig.write_image(Path(RESULTS_PATH_FOLDER, metric + "_split_down-ds.png"))

            # -- data group
            df_plot.sort_values(by=['theta'], inplace=True)
            t_means = df_plot.groupby('theta', as_index=False)[f"benefit_{metric}"].mean().to_dict()
            t_means = dict(zip(t_means['theta'].values(), t_means[f"benefit_{metric}"].values()))

            # -- box plot version
            fig = px.box(df_plot, y=f"benefit_{metric}", x='theta', color="method_label", color_discrete_map=COLOR_MAPS['method'], category_orders={'theta': df_plot.theta}, labels=COL_TO_LABEL)
            fig.update_layout(autosize=False, width=800, height=800, legend=layout_update_legend)
            fig.update_yaxes(dtick=.05, zerolinewidth=2)
            fig.update_xaxes(
                tickmode="array",
                tickvals=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
                ticktext=[f"{VAL_TO_LABEL[label.split('_')[0]]} ({round(t_means[label] * 100, 3):.2f}%)" for label in df_plot['theta'].unique()],
            )
            fig.write_image(Path(RESULTS_PATH_FOLDER, metric + "_enc-ds_dows-ds.png"))

            # -- architecture settings group
            df_plot.sort_values(by=['theta2_sorting'], inplace=True)
            t2_means = df_plot.groupby('theta2', as_index=False)[f"benefit_{metric}"].mean().to_dict()
            t2_means = dict(zip(t2_means['theta2'].values(), t2_means[f"benefit_{metric}"].values()))

            # -- variation as stack bar plot
            fig = px.box(df_plot, y=f"benefit_{metric}", x='theta2', color="method_label", color_discrete_map=COLOR_MAPS['method'], labels=COL_TO_LABEL, category_orders={'method_label': ["PAWS", "SimCLR", "SimSiam", "SimTriplet"]})
            fig.update_layout(autosize=False, width=1800, height=800, legend=layout_update_legend)
            fig.update_xaxes(
                tickmode="array",
                tickvals=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                ticktext=[f"{VAL_TO_LABEL[label]} ({round(t2_means[label] * 100, 3):.2f}%)" for label in df_plot['theta2'].unique()],
            )
            fig.write_image(Path(RESULTS_PATH_FOLDER, metric + "_architecture.png"))

            # -- variation as stack bar plot
            fig = px.box(df_plot, y=f"benefit_{metric}", x='theta2', color="method_label", color_discrete_map=COLOR_MAPS['method'], labels=COL_TO_LABEL)
            fig.update_layout(autosize=False, width=800, height=800, legend=layout_update_legend)
            fig.update_xaxes(
                tickmode="array",
                tickvals=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                ticktext=[f"{VAL_TO_LABEL[label]} ({round(t2_means[label] * 100, 3):.2f}%)" for label in df_plot['theta2'].unique()],
            )
            fig.write_image(Path(RESULTS_PATH_FOLDER, metric + "_architecture_square.png"))


def create_ece_plot(csv_path: str, result_path: str) -> None:
    """ Reproduce ECE Distribution Plot - Fig. A.1(g)
    """
    df = pd.read_csv(csv_path)
    # -- remove supevised runs
    df = df[~df['checkpoint_uuid'].isnull()]
    df.loc[df['freeze_encoder'] != 'True', "freeze_encoder"] = 'False'

    df['theta2'] = df['freeze_encoder'] + df['encoder'] + df['pred_head_structure']
    df['theta2_sorting'] = df['method'] + df['freeze_encoder'] + df['encoder'] + df['pred_head_structure']
    df.sort_values(by=['theta2_sorting'], inplace=True)

    fig = px.box(df, y='test_ece', x='theta2', color="method_label", color_discrete_map=COLOR_MAPS['method'], labels=COL_TO_LABEL)
    fig.update_layout(autosize=False, width=1400, height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title_text=''))
    fig.update_xaxes(
        tickmode="array",
        tickvals=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        ticktext=[f"{VAL_TO_LABEL[label]}" for label in df['theta2'].unique()],
    )
    fig.write_image(Path(result_path, "ece.png"))


def create_acc_plot(csv_path: str, result_path: str) -> None:
    """ Reproduce Maximum Accuracy Plot - Fig. 2 
    """
    df = pd.read_csv(csv_path)
    # -- fix faulty old PAWS configs, filter only finetuned runs, merge supervised runs
    df.loc[df['freeze_encoder'] != 'True', "freeze_encoder"] = 'False'
    df = df.loc[df['freeze_encoder'] != 'True']
    df.loc[df['method'].isin(['simclr-supervised', 'simsiam-supervised', 'simtriplet-supervised', 'paws-supervised']), 'method_label'] = 'Supervised'
    df_acc = df.groupby(['method_label', 'down_ds', 'split_size']).max().reset_index()

    # -- plot and cosmetics
    fig = px.line(df_acc, y='test_acc', x='split_size', color="method_label", color_discrete_map=COLOR_MAPS['method'], labels=COL_TO_LABEL, log_x=True, facet_col='down_ds', markers=True)
    fig.update_layout(autosize=False, width=2400, height=800, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title_text=''))
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.write_image(Path(result_path, "acc.png"))


def create_stability_table(csv_path: str, result_path: str) -> None:
    if not os.path.exists(Path(result_path)):
        os.makedirs(Path(result_path))
    df = pd.read_csv(csv_path)
    df = df.query("(pred_head_structure == 'one_layer_mlp') & (keep_enc_fc == True) & (encoder == 'resnet50')")

    with open(Path(result_path, 'table2.txt'), 'w') as table_file:
        table_file.write("Method & Acc (val) & Acc (test) & F1 (val) & F1 (test) \\\ \hline \n")
        for freeze_enc in ['True', '900']:
            for method in ['paws', 'simclr', 'simsiam', 'simtriplet']:
                df_tmp = df.query(f"(method == '{method}') & (freeze_encoder == '{freeze_enc}')")
                v_acc = list(df_tmp['v_acc'])
                f1_acc = list(df_tmp['v_f1'])
                test_acc = list(df_tmp['test_acc'])
                test_f1 = list(df_tmp['test_f1'])
                table_file.write(f"{VAL_TO_LABEL[method]} ({VAL_TO_LABEL[freeze_enc]}) & {round(np.mean(v_acc) * 100, 3):.2f}\% (+/- {round(np.std(v_acc)*100,3):.2f}) & {round(np.mean(test_acc) * 100, 3):.2f}\% (+/- {round(np.std(test_acc)*100,3):.2f})  & {round(np.mean(f1_acc) * 100, 3):.2f}\% (+/- {round(np.std(f1_acc)*100,3):.2f}) & {round(np.mean(test_f1) * 100, 3):.2f}\% (+/- {round(np.std(test_f1)*100,3):.2f}) \\\ \hline \n")


def reproduce_results():
    # -- reproduce paper figures
    csv_files = [('res/csv/split_reduce.csv', 'plots/split_reduce'), ('res/csv/split_reduce_abl-fc.csv', 'plots/split_reduce_alb-fc')]
    for file_path, result_path in csv_files:
        create_split_reduce_plots(file_path, result_path)
    create_ece_plot(*csv_files[0])
    create_acc_plot(*csv_files[0])
    create_stability_table('res/csv/stability.csv', 'table/encoder_sensivity')


if __name__ == '__main__':
    reproduce_results()
