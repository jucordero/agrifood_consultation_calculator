import streamlit as st
from utils.altair_plots import *
import matplotlib.pyplot as plt
from matplotlib import colors

from utils.consultation_utils import get_figure_captions

def paper_plots(datablock):

    st.markdown("# Paper plots")
    if (st.button("Update figure captions")):
        get_figure_captions.clear()

    captions = get_figure_captions()
    figs = []

    # ---------
    # Figure 1
    # ---------
    # FAOSTAT plot 

    to_plot = datablock["food"]["g/cap/day"].sel(Year=2050).fillna(0)
    to_plot["Item_group"].values = np.array(to_plot["Item_group"].values, dtype=str)
    to_plot = to_plot.fbs.group_sum(coordinate="Item_group", new_name="Item")

    to_plot = to_plot.rename({"food": "Retail"})
    to_plot = to_plot.rename({"production": "Production"})
    to_plot = to_plot.rename({"imports": "Imports"})
    to_plot = to_plot.rename({"exports": "Exports"})
    to_plot = to_plot.rename({"stock": "Stock"})
    to_plot = to_plot.rename({"losses": "Losses"})
    to_plot = to_plot.rename({"processing": "Processing"})
    to_plot = to_plot.rename({"other": "Other"})
    to_plot = to_plot.rename({"feed": "Feed"})
    to_plot = to_plot.rename({"seed": "Seed"})

    f1 = plot_bars_altair(to_plot, show="Item", x_axis_title="g/cap/day")
    figs.append(f1)

    # ---------
    # Figure 2
    # ---------
    # FAOSTAT plot weighted by kCal/cap/day 

    to_plot = datablock["food"]["kCal/cap/day"].sel(Year=2050).fillna(0)
    to_plot["Item_group"].values = np.array(to_plot["Item_group"].values, dtype=str)
    to_plot = to_plot.fbs.group_sum(coordinate="Item_group", new_name="Item")

    to_plot = to_plot.rename({"food": "Retail"})
    to_plot = to_plot.rename({"production": "Production"})
    to_plot = to_plot.rename({"imports": "Imports"})
    to_plot = to_plot.rename({"exports": "Exports"})
    to_plot = to_plot.rename({"stock": "Stock"})
    to_plot = to_plot.rename({"losses": "Losses"})
    to_plot = to_plot.rename({"processing": "Processing"})
    to_plot = to_plot.rename({"other": "Other"})
    to_plot = to_plot.rename({"feed": "Feed"})
    to_plot = to_plot.rename({"seed": "Seed"})

    f2 = plot_bars_altair(to_plot, show="Item", x_axis_title="kCal/cap/day")
    figs.append(f2)

    # ---------
    # Figure 3
    # ---------
    # Self-sufficiency plot

    to_plot = datablock["food"]["g/cap/day"].sel(Year=2050).fillna(0)
    to_plot["Item_group"].values = np.array(to_plot["Item_group"].values, dtype=str)
    to_plot = to_plot.fbs.group_sum(coordinate="Item_group", new_name="Item")

    to_plot["Domestic supply"] = (
        to_plot["production"] +
        to_plot["imports"] -
        to_plot["exports"] 
    )

    to_plot = to_plot.rename({"production": "Production"})


    f3 = plot_bars_altair2(
        to_plot,
        data_vars=["Production"],
        reversed_vars=["Domestic supply"], 
        show="Item",
        x_axis_title="g/cap/day",
        stacked=False,
    )
    figs.append(f3)

    # ---------
    # Figure 4
    # ---------
    # Self-sufficiency plot

    percentages_a = xr.DataArray(
        data=np.random.rand(3),
        dims=["Item"],
        coords={"Item": [
            "Agricultural emissions reduction",
            "Land use sinks",
            "Engineered removals"
            ]},
    )

    percentages_b = xr.DataArray(
        data=np.random.rand(3),
        dims=["Item"],
        coords={"Item": [
            "Agricultural emissions reduction",
            "Land use sinks",
            "Engineered removals"
            ]},
    )

    percentages_c = xr.DataArray(
        data=np.random.rand(3),
        dims=["Item"],
        coords={"Item": [
            "Agricultural emissions reduction",
            "Land use sinks",
            "Engineered removals"
            ]},
    )

    pctgs = xr.Dataset({
        "AFN+ a": percentages_a,
        "AFN+ b": percentages_b,
        "AFN+ c": percentages_c,
    })

    f4 = plot_bars_altair2(
        pctgs,
        data_vars=["AFN+ a", "AFN+ b", "AFN+ c"],
        reversed_vars=[],
        show="Item",
        x_axis_title="Proportion of net-zero target",
        stacked=False,
        horizontal=False,
    )

    figs.append(f4)

    # --------------------
    # Place figures in app
    # --------------------
    
    cols = st.columns((1,3,1))

    while len(figs) < len(captions):
        figs.append(None)
    with cols[1]:
        for fig_id, (f, capt) in enumerate(zip(figs, captions)):
            with st.expander(f"Figure {fig_id+1}"):
                if f is None:
                    st.write("In preparation")
                else:
                    st.altair_chart(f, use_container_width=True)
                st.caption(capt)
