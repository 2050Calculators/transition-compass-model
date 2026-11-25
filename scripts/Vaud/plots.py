import math

import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLOR_MODEL = "#2a78d6"
COLOR_OFFICIAL = "#eb6834"


def plot_comparaison(model, col_to_plot_vaud, legend_label, groups, dic_off_to_model):
    # Get the intersections of the model columns and dic_off_to_model values
    n_cols = 3
    if len(dic_off_to_model.values()) == 1:
        nb_rows = 1
    else:
        n_categories = len(set(model.columns) & set(dic_off_to_model.values()))
        # Determine subplot layout
        nb_rows = math.ceil(n_categories / n_cols)

    colors = {
        "Model": COLOR_MODEL,
        "Fribourg": COLOR_OFFICIAL,
    }

    # Filter years

    model.reset_index(inplace=True)
    model.rename(columns={"index": "Years"}, inplace=True)
    model = model[model["Years"].isin([2015, 2019, 2023])].reset_index(drop=True)

    # Sort groups: total should be plotted last
    groups_list = list(groups)

    groups_sorted = [g for g in groups_list if str(g[0]).lower() != "('total',)"] + [
        g for g in groups_list if str(g[0]).lower() == "('total',)"
    ]

    n_groups = len(groups_sorted)

    # Colors
    colors = {"Bilan Carbone": COLOR_OFFICIAL, "Transition Compass": COLOR_MODEL}

    # -------------------------
    # Main comparison figure
    # -------------------------

    fig = make_subplots(
        rows=nb_rows,
        cols=n_cols,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=[
            str(g[0][0]).capitalize()
            for g in groups_sorted
            if dic_off_to_model.get(g[0][0].lower()) in model.columns
        ],
    )

    # -------------------------
    # Ratio figure
    # -------------------------

    fig_ratio = go.Figure()

    bar_width = 0.1

    i = 0

    for name, group in groups_sorted:
        name = name[0].lower()

        if dic_off_to_model[name] not in model.columns:
            continue

        # Determine subplot position
        row = i // n_cols + 1
        col = i % n_cols + 1

        # Sort data
        group_sorted = (
            group[["Years", col_to_plot_vaud]]
            .sort_values("Years")
            .reset_index(drop=True)
        )

        years = group_sorted["Years"]
        vaud_values = group_sorted[col_to_plot_vaud]
        model_values = model[dic_off_to_model[name]]

        # -------------------------
        # Main comparison plot
        # -------------------------

        fig.add_trace(
            go.Scatter(
                x=years,
                y=vaud_values,
                mode="lines+markers",
                name="Bilan Carbone",
                legendgroup="Bilan Carbone",
                showlegend=(i == 0),
                line=dict(color=colors["Bilan Carbone"], dash="dash"),
                marker=dict(size=8),
                hovertemplate=("Année: %{x}<br>Bilan Carbone: %{y:.2f}<extra></extra>"),
            ),
            row=row,
            col=col,
        )

        fig.add_trace(
            go.Scatter(
                x=years,
                y=model_values,
                mode="lines+markers",
                name="Transition Compass",
                legendgroup="Transition Compass",
                showlegend=(i == 0),
                line=dict(color=colors["Transition Compass"], dash="solid"),
                marker=dict(size=8),
                hovertemplate=(
                    "Année: %{x}<br>Transition Compass: %{y:.2f}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

        # -------------------------
        # Ratio calculation
        # -------------------------

        ratio = (model_values - vaud_values) / vaud_values * 100

        # -------------------------
        # Ratio bar plot
        # -------------------------

        offset = (i - (n_groups - 1) / 2) * bar_width

        fig_ratio.add_trace(
            go.Bar(
                x=years + offset,
                y=ratio,
                name=name.capitalize(),
                width=bar_width,
                opacity=0.8,
                hovertemplate=(
                    "Année: %{x}<br>"
                    + name.capitalize()
                    + ": %{y:.2f}%"
                    + "<extra></extra>"
                ),
            )
        )

        if name == "total":
            print("Total ratio:", ratio)

        i += 1

    # -------------------------
    # Main figure styling
    # -------------------------

    fig.update_layout(
        height=360 * nb_rows,
        width=1100,
        template="plotly_white",
        title="Comparaison entre Transition Compass et le bilan carbone pour le canton de Vaud",
        font=dict(size=16),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5),
        margin=dict(t=100, b=70),
    )

    fig.update_xaxes(title_text="", showgrid=True, gridcolor="lightgrey")

    fig.update_yaxes(title_text="", showgrid=True, gridcolor="lightgrey")

    for col in range(1, n_cols):
        fig.update_yaxes(
            title_text="Année",
            row=nb_rows,
            col=col,
        )

    for row in range(1, nb_rows + 1):
        fig.update_yaxes(
            title_text=legend_label,
            row=row,
            col=1,
        )

    # -------------------------
    # Ratio figure styling
    # -------------------------

    fig_ratio.add_hline(y=0, line_color="black", line_width=1)

    fig_ratio.update_layout(
        template="plotly_white",
        width=1100,
        height=550,
        xaxis_title="Année",
        yaxis_title="Différence (%)",
        font=dict(size=16),
        barmode="overlay",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        margin=dict(r=200),
    )

    fig_ratio.update_xaxes(showgrid=True)
    fig_ratio.update_yaxes(showgrid=True, gridcolor="lightgrey")

    # -------------------------
    # Display figures
    # -------------------------

    fig.show()
    fig_ratio.show()

    return fig


def from_unit_to_other_unit_and_total(df, conv_factor=10**6):
    df["total"] = df.sum(axis=1)
    return df * conv_factor
