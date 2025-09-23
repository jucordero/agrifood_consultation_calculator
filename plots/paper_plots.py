import streamlit as st
from utils.altair_plots import *
import matplotlib.pyplot as plt
from matplotlib import colors

def paper_plots(datablock):

    cols = st.columns((1,3,1))
    with cols[1]:
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

        f = plot_bars_altair(to_plot, show="Item", x_axis_title="kCal/cap/day")
        st.altair_chart(f, use_container_width=True)
        st.caption("""**Figure 1**: Balance of food uses illustrated in broad categories showing
                (from top to bottom) i) UK production ii) UK imports iii) UK
                exports, iv) Total stock change v) losses associated to production
                vi) processing quantities associated to production vii) other uses
                viii) quantities used for animal feed, iv) quantities used for seeding
                x) Retail available quantities, typically for human consumption""")



    pass