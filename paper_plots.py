import os
import re
import glob

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import rc

fontstyle = {"family": "serif", "size": 13}
rc("font", **fontstyle)
rc("text", usetex=True)


def len_arr(x):
    if isinstance(x, list) or isinstance(x, np.ndarray):
        return len(x)
    else:
        return 1

def fname2arg(fname: str, arg: str):
    """Extract the number of layers from a filename"""
    pattern = rf"(?<={arg}_)[\d]*"
    result = re.search(pattern, fname)
    if result is not None:
        return int(result.group(0))
    return None
    

def get_data(
    ec_files,
    mc_files,
    exact_file,
):
    """Collect all data into a single dataframe for EC and MC. Collect ED data as well."""
    dfvec = []
    if ec_files is not None:
        for fname in ec_files:
            if os.path.isfile(fname):
                df_tmp = pd.read_pickle(fname)
                df_tmp["type"] = "EC"
                df_tmp["err"] = np.nan  # Append an empty column to make merges work
                if not "nlayer" in df_tmp.columns:
                    df_tmp["nlayer"] = fname2arg(fname, 'nlayer')
                if not "ncopy" in df_tmp.columns:
                    df_tmp["ncopy"] = fname2arg(fname, 'ncopy')
                dfvec.append(df_tmp)
    if mc_files is not None:
        for fname in mc_files:
            if os.path.isfile(fname):
                df_tmp = pd.read_pickle(fname)
                df_tmp["type"] = "MC"
                if not "nlayer" in df_tmp.columns:
                    df_tmp["nlayer"] = fname2arg(fname, 'nlayer')
                if not "ncopy" in df_tmp.columns:
                    df_tmp["ncopy"] = fname2arg(fname, 'ncopy')
                dfvec.append(df_tmp)
    df_mc_ec = pd.concat(dfvec)

    # Enrich dataset
    df_mc_ec["L"] = df_mc_ec["nx"].astype("str") + "x" + df_mc_ec["ny"].astype("str")
    df_mc_ec.rename(
        columns={
            "g2_el": "g_el",
            "g2_mag": "g_mag",
            "g2": "g",
        },
        inplace=True,
    )

    if "g" not in df_mc_ec.columns:
        df_mc_ec["g"] = df_mc_ec.g_el * 2
    if "g_mag" not in df_mc_ec.columns:
        df_mc_ec["g_mag"] = 1 / (df_tmp.g_el * 4)

    if exact_file is not None and os.path.isfile(exact_file):
        ed_data = pd.read_csv(exact_file)

    # different runs were done with versions of the code -
    # at some point the naming convention of Wilson loops was modified, we fix that here
    df_mc_ec.replace(
        {"wilson_00_11": "wilson_loop_0-0_1x1"},
        regex=True,
        inplace=True,
    )

    return df_mc_ec, ed_data


def select_data(
    df: pd.DataFrame,
    L: int,
    g_ints,
    g_masses,
    obs,
):
    """Helper function to select data"""
    cond = (
        (df["L"] == f"{L}x{L}")
        & (df["name"].isin(obs))
        & (df["g_mass"].isin(g_masses))
        & (df["g_int"].isin(g_ints))
    )
    data = df[cond]
    return data


def plot1_wilson(data, dest_dir, save_format):
    """Wilson loops"""

    df, _ = data

    # Process data
    df_2x2 = df[((df["L"] == "2x2") & (df["g"] <= 2.0))]
    df_2x2_wl = df_2x2[df_2x2["name"] == "wilson_loop_0-0_1x1"]
    df_2x2_wl_mass0 = df_2x2_wl[df_2x2_wl["g_mass"] == 0.0]

    df_large = df[((df["L"] == "6x6") & (df["g"] <= 2.0))]
    df_L_wl = df_large[df_large["name"] == "wilson_loop_0-0_1x1"]
    df_L_wl_mass0 = df_L_wl[df_L_wl["g_mass"] == 0.0]
    d6 = df_L_wl_mass0

    # Reformat data as needed for the heatmap
    data = df_2x2_wl_mass0.pivot(index="g_int", columns="g", values="mean").astype(
        float
    )

    # Plot
    f, ax = plt.subplots(1, 1)

    sns.heatmap(
        data,
        ax=ax,
        square=False,
        cbar_kws={"label": r"$1\times1$ Wilson Loop"},
    )
    ax.set_xticks([x + 0.5 for x in range(1, 21, 2)])
    ax.set_ylabel(r"$g_{I}$")
    ax.set_xlabel(r"$\lambda$")
    plt.savefig(os.path.join(dest_dir, "wilson_L2." + save_format))

    # Larger systems
    wl = 2  # wilson loop size
    f, axarr = plt.subplots(2, 1)
    palette = sns.color_palette("Paired", 2)
    colors = {1.0: palette[0], 0.5: palette[1]}
    styles = {1.0: "x", 0.0: "o"}

    for mass in [0.0, 1.0]:
        if mass == 0.0:
            ax = axarr[0]
        else:
            ax = axarr[1]

        d4 = df[
            (
                (df["L"] == "4x4")
                & (df["name"] == f"wilson_loop_0-0_{wl}x{wl}")
                & (df["g_mass"] == mass)
            )
        ]
        d6 = df[
            (
                (df["L"] == "6x6")
                & (df["name"] == f"wilson_loop_0-0_{wl}x{wl}")
                & (df["g_mass"] == mass)
            )
        ]

        for k, d in d4.groupby("g_int"):
            ax.scatter(
                d["g"],
                d["mean"],
                label=rf"$L=4, g_I={k}$",
                marker=styles[mass],
                s=25,
                color=colors[k],
            )

        for k, d in d6.groupby("g_int"):
            ax.scatter(
                d["g"],
                d["mean"],
                label=rf"$L=6, g_I={k}$",
                marker=styles[mass],
                s=25,
                color="red",
            )

    for ax in axarr:
        ax.set_ylabel(rf"${wl}\times{wl}$ Wilson Loop")
        ax.set_xlabel(r"$\lambda$")
        ax.legend()
    axarr[0].set_title("Mass = 0")
    axarr[1].set_title("Mass = 1")
    f.tight_layout()
    plt.savefig(
        os.path.join(
            dest_dir,
            "wilson_L4_L6." + save_format,
        )
    )

    return


def plot2_energy(data, dest_dir, save_format):
    """Energies Plot"""
    df, exact_data = data

    # Process data
    df_2x2_energy_mass0 = select_data(df, 2, [0.5, 1.0], [0.0], ["energy"])
    df_2x2_energy_mass1 = select_data(df, 2, [0.5, 1.0], [1.0], ["energy"])

    df_4x4_energy_mass0 = select_data(df, 4, [0.5, 1.0], [0.0], ["energy"])
    df_4x4_energy_mass1 = select_data(df, 4, [0.5, 1.0], [1.0], ["energy"])

    df_6x6_energy_mass0 = select_data(df, 6, [0.5, 1.0], [0.0], ["energy"])
    df_6x6_energy_mass1 = select_data(df, 6, [0.5, 1.0], [1.0], ["energy"])

    # Set up plot
    f, axarr = plt.subplots(2, 3)
    f.set_figheight(6)
    f.set_figwidth(15)

    palette = sns.color_palette("rocket", 2)
    colors = {1.0: palette[0], 0.5: palette[1]}

    # Plot ED data
    ed_data = exact_data[((exact_data["L"] == 2))]
    for g_mass, data in ed_data.groupby("g_mass"):
        ax = axarr[0, 0] if g_mass == 0.0 else axarr[1, 0]
        for g_int, d in data.groupby("g_int"):
            ax.plot(
                d["g"],
                d["energy"],
                c=colors[g_int],
            )
    ed_data = exact_data[exact_data["L"] == 4]
    for g_mass, data in ed_data.groupby("g_mass"):
        ax = axarr[0, 1] if g_mass == 0.0 else axarr[1, 1]
        for g_int, d in data.groupby("g_int"):
            ax.plot(
                d["g"],
                d["energy"],
                c=colors[g_int],
            )

    # Plot GGPEPS data
    # 2x2 data, mass = 0, top left
    for k, d in df_2x2_energy_mass0.groupby("g_int"):
        axarr[0][0].scatter(
            d["g"],
            d["mean"],
            label=k,
            s=20,
            c=d["g_int"].map(colors),
        )

    # 2x2 data, mass = 1, bottom left
    for k, d in df_2x2_energy_mass1.groupby("g_int"):
        axarr[1][0].scatter(
            d["g"],
            d["mean"],
            label=k,
            s=20,
            c=d["g_int"].map(colors),
        )

    # 4x4 data, mass = 0, top mid
    for k, d in df_4x4_energy_mass0.groupby("g_int"):
        g_int = d["g_int"].iloc[0]
        axarr[0][1].errorbar(
            d["g"],
            d["mean"],
            yerr=d["err"],
            label=k,
            fmt="o",
            markersize=5,
            c=colors[g_int],
        )

    # 4x4 data, mass = 1, bottom mid
    for k, d in df_4x4_energy_mass1.groupby("g_int"):
        g_int = d["g_int"].iloc[0]
        axarr[1][1].errorbar(
            d["g"],
            d["mean"],
            yerr=d["err"],
            label=k,
            fmt="o",
            markersize=5,
            c=colors[g_int],
        )

    # 6x6 data, mass = 0, top right
    for k, d in df_6x6_energy_mass0.groupby("g_int"):
        axarr[0][2].errorbar(
            d["g"],
            d["mean"],
            yerr=d["err"],
            label=k,
            fmt="o",
            markersize=5,
            c=colors[g_int],
        )

    # 6x6 data, mass = 1, bottom right
    for k, d in df_6x6_energy_mass1.groupby("g_int"):
        axarr[1][2].errorbar(
            d["g"],
            d["mean"],
            yerr=d["err"],
            label=k,
            fmt="o",
            markersize=5,
            c=colors[g_int],
        )

    # Axis labels
    axarr[0, 0].set_title("2x2 System", fontsize=15)
    axarr[0, 1].set_title("4x4 System", fontsize=15)
    axarr[0, 2].set_title("6x6 System", fontsize=15)
    axarr[0][0].set_ylabel("Energy")
    axarr[1][0].set_ylabel("Energy")
    axarr[0][1].set_ylabel("")
    axarr[1][1].set_ylabel("")
    axarr[0][2].set_ylabel("")
    axarr[1][2].set_ylabel("")
    axarr[0][0].set_xlabel("")
    axarr[0][1].set_xlabel("")
    axarr[0][2].set_xlabel("")
    axarr[1][0].set_xlabel(r"$\lambda$")
    axarr[1][1].set_xlabel(r"$\lambda$")
    axarr[1][2].set_xlabel(r"$\lambda$")

    # Fix the title in the legend
    for i, j in np.ndindex(axarr.shape):
        axarr[i, j].legend(title=r"$g_{I}$")

    plt.savefig(os.path.join(dest_dir, "energies." + save_format))
    return


def plot3_observables(data, dest_dir, save_format):
    """Observables Plot"""
    df, exact_data = data

    # Define labels for the plot
    obnames = {
        "energy": "energy",
        "el_energy": "electric energy",
        "mag_energy": "magnetic energy",
        "int_energy": "interaction energy",
        "mass_energy": "mass energy",
    }

    # Select data
    df_2x2_mass1_int1 = select_data(df, 2, [1.0], [1.0], obnames)
    ed_data = exact_data[
        (
            (exact_data["L"] == 2)
            & (exact_data["g_int"] == 1.0)
            & (exact_data["g_mass"] == 1.0)
        )
    ]

    # Plot
    colors = sns.color_palette("viridis", n_colors=len(obnames.keys()))
    f, ax = plt.subplots(1, 1)
    for ob, col in zip(obnames.keys(), colors):
        data = df_2x2_mass1_int1[df_2x2_mass1_int1["name"] == ob]

        sns.scatterplot(
            data=data,
            x="g",
            y="mean",
            alpha=0.9,
            ax=ax,
            label=obnames[ob],
            color=col,
        )

        sns.lineplot(
            data=ed_data,
            x="g",
            y=ob,
            alpha=0.9,
            ax=ax,
            label=None,
            color=col,
        )

    ax.set_ylabel("Energy")
    ax.set_xlabel(r"$\lambda$")
    plt.savefig(os.path.join(dest_dir, "observables." + save_format))
    return


def plot4_fermions(data, dest_dir, save_format):
    """Free fermions"""

    # hard coded data for free fermions
    sizes = [2 * k for k in range(1, 11)]
    free_fermion_gs = [
        -1.414213562,
        -1.0,
        -0.9683085158,
        -0.9619397663,
        -0.959948371,
        -0.959132462,
        -0.9587350656,
        -0.9585176647,
        -0.9583884742,
        -0.9583067892,
    ]

    f, ax = plt.subplots(1, 1)
    ax.plot(
        sizes,
        free_fermion_gs,
        marker="o",
        linestyle="--",
        color=(22 / 256, 170 / 256, 162 / 256),
    )
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"Ground State Energy / $L^2$")
    ax.xaxis.set_ticks(sizes)
    ax.grid(axis="both", alpha=0.2)
    plt.savefig(
        os.path.join(
            dest_dir,
            "free_fermions." + save_format,
        )
    )
    return


def main():

    # Data
    ec_files = glob.glob(r"data/L_*/ec/mass_*/g_*_int_*/summary*.pkl")
    mc_files = glob.glob(r"data/L_*/mc/mass_*/g_*_int_*_mass_*/summary*.pkl")
    exact_file = r"data/ed_data.csv"
    data = get_data(ec_files, mc_files, exact_file)

    # Generate plots
    save_format = "pdf"  # or png, etc.
    dest_dir = "generated_plots"
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    plot1_wilson(data, dest_dir, save_format)
    plot2_energy(data, dest_dir, save_format)
    plot3_observables(data, dest_dir, save_format)
    plot4_fermions(data, dest_dir, save_format)
    return


if __name__ == "__main__":

    main()
