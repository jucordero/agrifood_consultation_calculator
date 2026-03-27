import xarray as xr
from .glossary import sector_emissions_dict

def replace_coord_labels(da, coord_name, label_dict):
    """Replaces the labels of a coordinate in a DataArray according to a provided dictionary."""
    new_labels = [label_dict.get(label, label) for label in da.coords[coord_name].values]
    da = da.assign_coords({coord_name: new_labels})
    return da

def merge_coord_labels(da, coord_name, label_dict):
    """Merges the labels of a coordinate in a DataArray according to a provided dictionary."""
    preserved_coords = {
        name: coord
        for name, coord in da.coords.items()
        if name != coord_name and coord.dims and coord_name not in coord.dims
    }
    new_labels = [label_dict.get(label, label) for label in da.coords[coord_name].values]
    da = da.assign_coords({coord_name: new_labels})
    da = da.groupby(coord_name).sum()
    if preserved_coords:
        da = da.assign_coords(preserved_coords)
    return da

def compute_metrics(
        datablock
        ):
    """Computes a series of metrics from the resulting datablock"""
    
    print("Computing metrics...")

    metric_yr = 2050
    datablock["metrics"] = {}

    years_array = datablock["food"]["g/cap/day"]["production"].Year.values

    # nutritional_values
    qty_keys = ["g_prot/cap/day", "g_fat/cap/day", "kCal/cap/day"]
    nutrition_keys = ["g_prot/g_food", "g_fat/g_food", "kCal/g_food"]

    for qk, nk in zip(qty_keys, nutrition_keys):
        datablock["food"][qk] = datablock["food"][nk] * datablock["food"]["g/cap/day"]

    # Emissions balance
    # -----------------

    reference_emissions_baseline = datablock["advanced_settings"]["baseline_total_emissions"]
    reference_emissions_baseline_agriculture = datablock["advanced_settings"]["baseline_agricultural_emissions"]

    # seq_da = datablock["impact"]["co2e_sequestration"].sel(Year=metric_yr)
    seq_da = datablock["impact"]["co2e_sequestration"]
    
    # agriculture_emissions = datablock["impact"]["g_co2e/year"]["production"].sel(Year=metric_yr)/1e6
    agriculture_emissions = datablock["impact"]["g_co2e/year"]["production"]/1e6
    total_agriculture_emissions = agriculture_emissions.sum(dim="Item")/1e6
    total_seq = seq_da.sel(Item=["Broadleaf woodland",
                                 "Coniferous woodland",
                                 "New Broadleaf woodland",
                                 "New Coniferous woodland",
                                 "Managed pasture",
                                 "Managed arable",
                                 "Mixed farming",
                                 "Silvopasture",
                                 "Agroforestry",
                                 "Bioenergy crops (arable)",
                                 "Bioenergy crops (pasture)",
                                 ]).sum(dim="Item")/1e6

    # land_use_emissions = datablock["impact"]["g_co2e/year_land"]["production"].sel(Year=metric_yr)/1e6
    land_use_emissions = datablock["impact"]["g_co2e/year_land"]["production"]/1e6
    total_land_use_emissions = land_use_emissions.sum(dim="Item")/1e6

    total_removals = seq_da.sel(Item=["BECCS from waste",
                                      "BECCS from overseas biomass",
                                      "BECCS from land",
                                      "DACCS",
                                      "Biochar"]).sum(dim="Item")/1e6
    
    # emissions_balance = xr.DataArray(data = list(sector_emissions_dict.values()),
    #                         name="Sectoral emissions",
    #                         coords={"Sector": list(sector_emissions_dict.keys())})
    
    # emissions_balance = emissions_balance.expand_dims(Year=years_array).copy()

    emissions_balance = datablock["balanced_pathway"]
    emissions_balance = emissions_balance.reindex(Year=list(range(2020, 2051)), fill_value=None).fillna(emissions_balance.sel(Year=2025))

    emissions_balance = emissions_balance.drop_sel(Sector="Land use")

    emissions_balance = replace_coord_labels(
        emissions_balance,
        "Sector",
        {
            "Engineered removals": "Removals",
            "Industry": "Manufacturing",
            "Electricity supply": "Electricity",
            "Surface transport": "Transport",
        }
        )
    
    emissions_balance = merge_coord_labels(
        emissions_balance,
        "Sector",
        {
            "Residential buildings": "Buildings",
            "Non-residential buildings": "Buildings",
        }
        )

    emissions_balance = emissions_balance.assign_coords(
        Sector = emissions_balance.Sector.astype(str).where(emissions_balance.Sector != "Engineered removals", "Removals"))
    
    emissions_balance = xr.concat([emissions_balance, xr.DataArray(
        data=[[6.08]*len(years_array), [0]*len(years_array)],
        coords={"Sector": ["LU sources", "LU sinks"], "Year": emissions_balance.Year.values},
        dims=["Sector", "Year"]
    )], dim="Sector")


    emissions_balance.loc[{"Sector": "Removals"}] = -9.82

    emissions_balance.loc[{"Sector": "Agriculture"}] = total_agriculture_emissions
    emissions_balance.loc[{"Sector": "LU sinks"}] -= total_seq
    emissions_balance.loc[{"Sector": "Removals"}] -= total_removals

    emissions_balance.loc[{"Sector": "LU sources"}] -= seq_da.sel(Item=["Restored upland peat", "Restored lowland peat"]).sum(dim="Item").values/1e6
    emissions_balance.loc[{"Sector": "LU sources"}] += total_land_use_emissions

    total_emissions = emissions_balance.sum(dim="Sector")

    reducion_emissions_pctg = (total_emissions - reference_emissions_baseline) / reference_emissions_baseline * 100

    forest_sequestration_MtCO2 = seq_da.sel(Item=["Broadleaf woodland", "Coniferous woodland"]).sum(dim="Item").values/1e6

    agricultural_emissions = emissions_balance.sel(Sector="Agriculture")
    reduction_emissions_agricultural_pctg = (agricultural_emissions - reference_emissions_baseline_agriculture) / reference_emissions_baseline_agriculture * 100

    datablock["metrics"]["emissions_balance"] = emissions_balance
    datablock["metrics"]["total_sequestration"] = total_seq
    datablock["metrics"]["total_removals"] = total_removals
    datablock["metrics"]["total_emissions"] = total_emissions
    datablock["metrics"]["reference_emissions_baseline"] = reference_emissions_baseline
    datablock["metrics"]["reduction_emissions_pctg"] = reducion_emissions_pctg
    datablock["metrics"]["forest_sequestration_MtCO2"] = forest_sequestration_MtCO2
    datablock["metrics"]["agricultural_emissions"] = agricultural_emissions
    datablock["metrics"]["reduction_emissions_agricultural_pctg"] = reduction_emissions_agricultural_pctg

    # SSR
    # ---
    ssr_metric = datablock["advanced_settings"]["ssr_metric"]
    
    # gcapday = datablock["food"][ssr_metric].sel(Year=metric_yr).fillna(0)
    gcapday = datablock["food"][ssr_metric].fillna(0)
    gcapday = gcapday.fbs.group_sum(coordinate="Item_origin", new_name="Item")
    
    # gcapday_ref = datablock["food"][ssr_metric].sel(Year=2020).fillna(0)
    gcapday_ref = datablock["food"][ssr_metric].fillna(0)
    gcapday_ref = gcapday_ref.fbs.group_sum(coordinate="Item_origin", new_name="Item")

    SSR_ref = gcapday_ref.fbs.SSR()
    SSR_metric_yr = gcapday.fbs.SSR()

    datablock["metrics"]["SSR_ref"] = SSR_ref
    datablock["metrics"]["SSR_metric_yr"] = SSR_metric_yr
    datablock["metrics"]["gcapday_item_origin"] = gcapday
    datablock["metrics"]["gcapday_ref_item_origin"] = gcapday_ref

    # Herd size

    # Read baseline herd sizes from session state
    baseline_beef_herd = datablock["advanced_settings"]["baseline_beef_herd"]
    baseline_dairy_herd = datablock["advanced_settings"]["baseline_dairy_herd"]
    baseline_poultry_heads = datablock["advanced_settings"]["baseline_poultry_heads"]
    baseline_pig_heads = datablock["advanced_settings"]["baseline_pig_heads"]
    baseline_sheep_flock = datablock["advanced_settings"]["baseline_sheep_flock"]
    baseline_dairy_herd_2y = datablock["advanced_settings"]["baseline_dairy_herd_breeding_aged_2_years_"]

    dairy_herd_beef = datablock["advanced_settings"]["dairy_herd_beef"]
    # Read total population from datablock
    pop_baseline = datablock["population"]["population"].sel(Region = 826, Year=2020)
    pop_new = datablock["population"]["population"].sel(Region = 826)

    # Dairy herd

    baseline_dairy_production = pop_baseline * datablock["food"]["g/cap/day"]["production"].sel(Year=2020, Item=[2743, 2740, 2948]).fillna(0).sum()
    new_dairy_production = pop_new * datablock["food"]["g/cap/day"]["production"].sel(Item=[2743, 2740, 2948]).fillna(0).sum(dim="Item")
    new_dairy_herd = new_dairy_production / baseline_dairy_production * baseline_dairy_herd
    new_dairy_herd["Item"] = "Dairy herd"
    new_dairy_herd.name = "Dairy herd"
    new_dairy_herd_2y = new_dairy_production / baseline_dairy_production * baseline_dairy_herd_2y
    new_dairy_herd_2y["Item"] = "Dairy herd 2 years and older"
    new_dairy_herd_2y.name = "Dairy herd 2 years and older"

    datablock["metrics"]["baseline_dairy_herd"] = baseline_dairy_herd
    datablock["metrics"]["new_dairy_herd"] = new_dairy_herd
    datablock["metrics"]["new_dairy_herd_2y"] = new_dairy_herd_2y

    # Beef herd
    baseline_beef_production = pop_baseline * datablock["food"]["g/cap/day"]["production"].sel(Year=2020, Item=2731).fillna(0).sum()
    new_beef_production = pop_new * datablock["food"]["g/cap/day"]["production"].sel(Item=2731).fillna(0)
    new_beef_herd = baseline_beef_herd * (new_beef_production - dairy_herd_beef * baseline_beef_production * new_dairy_herd / baseline_dairy_herd) / ((1 - dairy_herd_beef)*baseline_beef_production)
    new_beef_herd["Item"] = "Beef herd"
    new_beef_herd.name = "Beed herd"

    datablock["metrics"]["baseline_beef_herd"] = baseline_beef_herd
    datablock["metrics"]["new_beef_herd"] = new_beef_herd
    datablock["metrics"]["new_herd"] = new_dairy_herd + new_beef_herd

    # Poultry, pigs and sheep
    baseline_poultry_production = pop_baseline * datablock["food"]["g/cap/day"]["production"].sel(Year=2020, Item=2734).fillna(0).sum()
    new_poultry_production = pop_new * datablock["food"]["g/cap/day"]["production"].sel(Item=2734).fillna(0)
    new_poultry_heads = baseline_poultry_heads * new_poultry_production / baseline_poultry_production
    new_poultry_heads["Item"] = "Poultry heads"
    new_poultry_heads.name = "Poultry heads"

    datablock["metrics"]["baseline_poultry_heads"] = baseline_poultry_heads
    datablock["metrics"]["new_poultry_heads"] = new_poultry_heads

    baseline_pig_production = pop_baseline * datablock["food"]["g/cap/day"]["production"].sel(Year=2020, Item=2733).fillna(0).sum()
    new_pig_production = pop_new * datablock["food"]["g/cap/day"]["production"].sel(Item=2733).fillna(0)
    new_pig_heads = baseline_pig_heads * new_pig_production / baseline_pig_production
    new_pig_heads["Item"] = "Pig heads"
    new_pig_heads.name = "Pig heads"

    datablock["metrics"]["baseline_pig_heads"] = baseline_pig_heads
    datablock["metrics"]["new_pig_heads"] = new_pig_heads

    baseline_sheep_production = pop_baseline * datablock["food"]["g/cap/day"]["production"].sel(Year=2020, Item=2732).fillna(0).sum()
    new_sheep_production = pop_new * datablock["food"]["g/cap/day"]["production"].sel(Item=2732).fillna(0)
    new_sheep_flock = baseline_sheep_flock * new_sheep_production / baseline_sheep_production
    new_sheep_flock["Item"] = "Sheep flock"
    new_sheep_flock.name = "Sheep flock"

    datablock["metrics"]["baseline_sheep_flock"] = baseline_sheep_flock
    datablock["metrics"]["new_sheep_flock"] = new_sheep_flock

    size_dataarrays = [new_dairy_herd, new_dairy_herd_2y, new_beef_herd,
                       new_poultry_heads, new_pig_heads, new_sheep_flock]

    xr.set_options(use_new_combine_kwarg_defaults=True)

    for da in size_dataarrays:
        if "Item" not in da.dims:
            da = da.expand_dims(dim="Item")

        # Add to datablock
        if "livestock" not in datablock["metrics"]:
            datablock["metrics"]["livestock"] = da
        else:
            datablock["metrics"]["livestock"] = xr.concat([datablock["metrics"]["livestock"], da], dim="Item")

    # Land use
    pctg = datablock["land"]["percentage_land_use"]
    totals = pctg.sum(dim=["x", "y"])

    total_pasture = totals.sel(aggregate_class=["Improved grassland",
                                                "Semi-natural grassland",
                                                "Managed pasture",
                                                "Silvopasture"]).sum().values

    baseline_pasture = datablock["land"]["baseline"].sel(aggregate_class=["Improved grassland",
                                                                          "Semi-natural grassland"]).sum().values

    total_forest = totals.sel(aggregate_class=["Broadleaf woodland",
                                               "Coniferous woodland",
                                               "New Broadleaf woodland",
                                               "New Coniferous woodland"]).sum().values

    new_forest_land = (total_forest - datablock["land"]["baseline"].sel(aggregate_class=["Broadleaf woodland", "Coniferous woodland"]).sum().values)

    baseline_forest = datablock["land"]["baseline"].sel(aggregate_class=["Broadleaf woodland",
                                                                         "Coniferous woodland"]).sum().values

    total_arable = totals.sel(aggregate_class=["Arable",
                                               "Managed arable",
                                               "Mixed farming",
                                               "Agroforestry"]).sum().values

    total_agroforestry = totals.sel(aggregate_class="Agroforestry").sum().values
    total_silvopasture = totals.sel(aggregate_class="Silvopasture").sum().values
    total_mixed_farming = totals.sel(aggregate_class="Mixed farming").sum().values
    total_beccs = totals.sel(aggregate_class=["Bioenergy crops (pasture)", "Bioenergy crops (arable)"]).sum().values

    if datablock["run_params"]["land_BECCS_pasture"] + datablock["run_params"]["land_BECCS"] != 0:
        beccs_on_pasture = total_beccs * datablock["run_params"]["land_BECCS_pasture"] / (datablock["run_params"]["land_BECCS_pasture"] + datablock["run_params"]["land_BECCS"])
        beccs_on_arable = total_beccs * datablock["run_params"]["land_BECCS"] / (datablock["run_params"]["land_BECCS_pasture"] + datablock["run_params"]["land_BECCS"])
    else:
        beccs_on_pasture = 0
        beccs_on_arable = 0

    baseline_arable = datablock["land"]["baseline"].sel(aggregate_class=["Arable"]).sum().values

    new_arable_land_pctg = (total_arable - baseline_arable) / baseline_arable * 100
    new_pasture_land_pctg = (total_pasture - baseline_pasture) / baseline_pasture * 100

    total_restored_peatland = totals.sel(aggregate_class=["Restored upland peat", "Restored lowland peat"]).sum().values

    datablock["metrics"]["total_restored_peatland"] = total_restored_peatland
    datablock["metrics"]["total_pasture"] = total_pasture
    datablock["metrics"]["total_forest"] = total_forest
    datablock["metrics"]["total_arable"] = total_arable
    datablock["metrics"]["baseline_pasture"] = baseline_pasture
    datablock["metrics"]["baseline_forest"] = baseline_forest
    datablock["metrics"]["baseline_arable"] = baseline_arable
    datablock["metrics"]["new_forest_land"] = new_forest_land
    datablock["metrics"]["new_arable_land_pctg"] = new_arable_land_pctg
    datablock["metrics"]["new_pasture_land_pctg"] = new_pasture_land_pctg

    datablock["metrics"]["total_agroforestry"] = total_agroforestry
    datablock["metrics"]["total_silvopasture"] = total_silvopasture
    datablock["metrics"]["total_mixed_farming"] = total_mixed_farming
    datablock["metrics"]["total_beccs"] = total_beccs
    datablock["metrics"]["beccs_on_pasture"] = beccs_on_pasture
    datablock["metrics"]["beccs_on_arable"] = beccs_on_arable

    # Crop sizes

    gcapday = datablock["food"]["g/cap/day"]["production"]

    baseline_potatoes_area_mha = datablock["advanced_settings"]["baseline_potato_area"]
    baseline_potato_production = pop_baseline * gcapday.sel(Year=2020, Item=2531).fillna(0).sum().values
    new_potato_production = pop_new * gcapday.sel(Year=metric_yr, Item=2531).fillna(0).sum().values
    new_potato_area = baseline_potatoes_area_mha * new_potato_production / baseline_potato_production
    datablock["metrics"]["new_potato_area"] = new_potato_area


    baseline_oilseed_area_mha = datablock["advanced_settings"]["baseline_oilseed_area"]
    baseline_oilseed_production = pop_baseline * gcapday.sel(Year=2020, Item=[2570, 2572, 2573, 2575, 2576, 2577, 2578, 2579, 2581, 2582, 2586 ]).fillna(0).sum().values
    new_oilseed_production = pop_new * gcapday.sel(Year=metric_yr, Item=[2570, 2572, 2573, 2575, 2576, 2577, 2578, 2579, 2581, 2582, 2586 ]).fillna(0).sum().values
    new_oilseed_area = baseline_oilseed_area_mha * new_oilseed_production / baseline_oilseed_production
    datablock["metrics"]["new_oilseed_area"] = new_oilseed_area

    baseline_cereal_area_mha = datablock["advanced_settings"]["baseline_cereal_area"]
    baseline_cereal_production = pop_baseline * gcapday.sel(Year=2020, Item=gcapday.Item_group=="Cereals - Excluding Beer").fillna(0).sum().values
    new_cereal_production = pop_new * gcapday.sel(Year=metric_yr, Item=gcapday.Item_group=="Cereals - Excluding Beer").fillna(0).sum().values

    new_cereal_area = baseline_cereal_area_mha * new_cereal_production / baseline_cereal_production
    datablock["metrics"]["new_cereal_area"] = new_cereal_area

    baseline_horticulture_area_mha = datablock["advanced_settings"]["baseline_horticulture_area"]
    new_horiticulture_area = baseline_horticulture_area_mha * total_arable / baseline_arable * (1+datablock["run_params"]["horticulture"]/100)
    datablock["metrics"]["new_horticulture_area"] = new_horiticulture_area

    other_crops_area_mha = total_arable/1e6 - new_potato_area - new_oilseed_area - new_cereal_area - new_horiticulture_area
    datablock["metrics"]["other_crops_area_mha"] = other_crops_area_mha

    # Food balance sheet

    population = datablock["population"]["population"].sel(Region=826)
    food_qty = datablock["food"]["g/cap/day"]

    datablock["food"]["kton/year"] = food_qty * population / 1e6 * 365.25

    return datablock