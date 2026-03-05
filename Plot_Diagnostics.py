import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
from cartopy import config
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from pathlib import Path
import sys
from matplotlib import ticker,_cm
from datetime import datetime, timedelta
import cftime
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from dateutil import tz
import gsw
from matplotlib.ticker import LogFormatter, LogLocator
import matplotlib.ticker as ticker
import scienceplots
import cmasher as cmr
import cmocean as cmo
plt.style.use('science')

class Diagnostics:
    def __init__(self,run_dir,start_date):

        self.run_dir = Path(run_dir)
        self.folder_path = self.run_dir / "plots"
        self.folder_path.mkdir(exist_ok=True)

        self.ds = nc.Dataset(self.run_dir / "global_1deg.averages.nc")
        self.ds_energy = nc.Dataset(self.run_dir / "global_1deg.energy.nc")

        self.xt = self.ds.variables["xt"][:]
        self.xt, self.sort_idx = self.cyclic(self.xt)
        self.xu = self.ds.variables['xu'][:]
        self.xu, self.sort_idx_u = self.cyclic(self.xu)
        self.yt = self.ds.variables["yt"][:]
        self.yu = self.ds.variables['yu'][:]
        self.zt = self.ds.variables["zt"][:]
        self.zw = self.ds.variables["zw"][:]

        units = f"days since {start_date:%Y-%m-%d}"

        time_var = self.ds.variables["Time"]
        self.dates = cftime.num2date(
            time_var[:],
            units=units,
            calendar="360_day"
        )
        
        time_var_nrj = self.ds_energy["Time"] 
        self.nrj_dates = cftime.num2date(
            time_var_nrj[:],
            units=units,
            calendar="360_day"
        )        

        self.epsilon = 1e-6

    def get_time_indices_old(self, n_years=10, use_energy=False):
        dates = self.nrj_dates if use_energy else self.dates
        last_year = dates[-1].year
        first_year = last_year - n_years + 1
        idx = [i for i, d in enumerate(dates) if d.year >= first_year]
        return np.array(idx)

    def get_time_indices(self, n_years=10, use_energy=False):
        dates = self.nrj_dates if use_energy else self.dates
        last_date = dates[-1]
        if last_date.month < 12 or last_date.day < 30:
            last_year = last_date.year - 1
        else:
            last_year = last_date.year
    
        first_year = last_year - n_years + 1
    
        idx = [i for i, d in enumerate(dates) if first_year <= d.year <= last_year]
    
        return np.array(idx)

    def cyclic(self,x):
        x = ((x + 180) % 360) -180
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        return x,sort_idx

    def plot_map(self, lon, lat, field, title, filename,
                 cmap="coolwarm", levels=150,
                 u=None, v=None):

        vmin = np.nanmin(field)
        vmax = np.nanmax(field)+self.epsilon

        plt.figure(figsize=(15, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())

        cf = plt.contourf(
            lon, lat, field,
            np.linspace(vmin, vmax, levels),
            cmap=cmap,
            transform=ccrs.PlateCarree(),
            extend="both"
        )
        cf.set_edgecolor('face')

        plt.colorbar(cf, fraction=0.046, pad=0.04)

        if u is not None and v is not None:
            ax.streamplot(
                lon, lat, u, v,
                color="w",
                linewidth=0.3,
                density=1.5,
                transform=ccrs.PlateCarree()
            )

        ax.add_feature(cfeature.GSHHSFeature("low", levels=[1, 2, 6]))
        ax.coastlines()

        plt.title(title, fontsize=18)
        plt.savefig(self.folder_path / filename,
                    dpi=200, bbox_inches="tight")
        plt.close()

    def vertical_plot_map(self, lat, depth, field, title, filename,
                          cmap="coolwarm", levels=150):
    
        vmin = np.nanmin(field)
        vmax = np.nanmax(field) + self.epsilon
    
        plt.figure(figsize=(10, 6))
    
        cf = plt.contourf(
                lat, -depth, field,
            np.linspace(vmin, vmax, levels),
            cmap=cmap,
            extend="both"
        )
    
        cf.set_edgecolor("face")
    
        plt.colorbar(cf, fraction=0.046, pad=0.04)
    
        plt.gca().invert_yaxis()  # depth increasing downward
    
        plt.xlabel("Latitude")
        plt.ylabel("Depth (m)")
        plt.title(title, fontsize=18)
    
        plt.savefig(
            self.folder_path / filename,
            dpi=200,
            bbox_inches="tight"
        )
    
        plt.close()

    def temp(self, n_years=10,vert=False):
        idx = self.get_time_indices(n_years)
        year = self.dates[idx[-1]].year
        cmap = cmo.cm.thermal
        if vert:
            temp = np.mean(
                self.ds.variables["temp"][idx, :, :, :],
                axis=(0,3)
            )
            self.vertical_plot_map(
                 self.yt, self.zt, temp,
                f"Temperature (last {n_years} years)",
                f"Temp_{year}_vert.pdf",
                cmap = cmap
            )

        else:
            temp = np.mean(
                self.ds.variables["temp"][idx, -1, :, :],
                axis=0
            )

            temp = temp[:, self.sort_idx]


            self.plot_map(
                self.xt, self.yt, temp,
                f"SST (last {n_years} years)",
                f"SST_{year}.pdf",
                cmap = cmap
            )
        if vert:
            print("SST done")
        else:
            print("Vertical temperature done")

    def salt(self, n_years=10,vert=False):
        idx = self.get_time_indices(n_years)
        year = self.dates[idx[-1]].year
        cmap = cmo.cm.haline
        if vert:
            salt = np.mean(
                self.ds.variables["salt"][idx, :, :, :],
                axis=(0,3)
            )
            self.vertical_plot_map(
                 self.yt, self.zt, salt,
                f"Salinity (last {n_years} years)",
                f"Salt_{year}_vert.pdf",
                cmap = cmap
            )

        else:
            salt = np.mean(
                self.ds.variables["salt"][idx, -1, :, :],
                axis=0
            )

            salt = salt[:, self.sort_idx]


            self.plot_map(
                self.xt, self.yt, salt,
                f"SSS (last {n_years} years)",
                f"SSS_{year}.pdf",
                cmap = cmap
            )

        if vert:
            print("SSS done")
        else:
            print("Vertical salinity done")

    def ssh(self, n_years=10):
        idx = self.get_time_indices(n_years)
        ssh = np.mean(
            self.ds.variables["ssh"][idx, :, :],
            axis=0
        )
        ssh = ssh[:, self.sort_idx]
        ssh = ssh - np.nanmean(ssh)
        year = self.dates[idx[-1]].year
        self.plot_map(
            self.xt, self.yt, ssh,
            f"SSH (last {n_years} years)",
            f"SSH_{year}.pdf"
        )
        print("SSH done")

    def heat_flux(self, n_years=10):

        idx = self.get_time_indices(n_years)

        qnet = np.mean(
            self.ds.variables["qnet"][idx, :, :],
            axis=0
        )

        qsol = np.mean(
            self.ds.variables["qsol"][idx, :, :],
            axis=0
        )

        Q = (qnet + qsol)[:, self.sort_idx]
        year = self.dates[idx[-1]].year

        self.plot_map(
            self.xt, self.yt, Q,
            f"Heat flux (last {n_years} years)",
            f"HeatFlux_{year}.pdf"
        )

        print("Heat flux done")

    def compute_mld(self,temp,salt,zt,drho=0.03, zref=10.0):
        nz, ny, nx = temp.shape
        kref = np.argmin(np.abs(zt - zref))
        rho = gsw.rho(salt, temp, 0)
        rho_ref = rho[kref, :, :]
        mld = np.full((ny, nx), np.nan)

        for k in range(kref, - 1, -1):
            mask = (rho[k, :, :] - rho_ref) >= drho
            newly_found = mask & np.isnan(mld)
            mld[newly_found] = zt[k]
    
        return -mld

    def seasonal_mld(self, n_years=10):
    
        idx = self.get_time_indices(n_years=n_years)
    
        zt = self.zt
        lat = self.yt
    
        idx_winter = [i for i in idx if self.dates[i].month in [1, 2, 3]]
        idx_summer = [i for i in idx if self.dates[i].month in [7, 8, 9]]
    
        mld_winter_all = []
        for it in idx_winter:
            mld_winter_all.append(
                self.compute_mld(
                    self.ds.variables['temp'][it, :, :, :][:, :, self.sort_idx],
                    self.ds.variables['salt'][it, :, :, :][:, :, self.sort_idx],
                    zt
                )
            )
    
        mld_summer_all = []
        for it in idx_summer:
            mld_summer_all.append(
                self.compute_mld(
                    self.ds.variables['temp'][it, :, :, :][:, :, self.sort_idx],
                    self.ds.variables['salt'][it, :, :, :][:, :, self.sort_idx],
                    zt
                )
            )
    
        mld_march = np.mean(mld_winter_all, axis=0)
        mld_sept = np.mean(mld_summer_all, axis=0)
    
        north = lat > 0
        north2d = north[:, None]
    
        # Hemisphere logic
        mld_winter = np.where(north2d, mld_march, mld_sept)
        mld_summer = np.where(north2d, mld_sept, mld_march)
    
        return mld_winter, mld_summer

    def mld(self, n_years=10):
    
        mld_winter, mld_summer = self.seasonal_mld(n_years)
    
        maxvals = [500,100] #[np.nanmax(mld_winter), np.nanmax(mld_summer)]
        scales = ['linear','linear']
        fig, axs = plt.subplots(
            1, 2,
            figsize=(18, 6),
            subplot_kw={'projection': ccrs.PlateCarree()}
        )
    
        titles = [
            f'Winter MLD (last {n_years} years)',
            f'Summer MLD (last {n_years} years)'
        ]
    
        fields = [mld_winter, mld_summer]
    
        for ax, field, title, vmax, scale in zip(axs, fields, titles, maxvals, scales):
            if scale == 'log':
                norm = mcolors.LogNorm(vmin=1, vmax=vmax)
                levels = np.logspace(np.log10(1), np.log10(vmax), num=200)

            else:
                levels = np.arange(0, vmax, vmax/200)
                norm = mcolors.Normalize(vmin=0, vmax=vmax)

            cf = ax.contourf(
                self.xt, self.yt, field,
                levels=levels,
                cmap='viridis',
                norm=norm,
                extend='max',
                transform=ccrs.PlateCarree()
            )

            cf.set_edgecolor('face')
            ax.add_feature(cfeature.GSHHSFeature('low', levels=[1, 2, 6]))
            ax.coastlines()
            ax.set_title(title, fontsize=16)
        
            cbar = fig.colorbar(
                cf,
                ax=ax,
                orientation='horizontal',
                fraction=0.06,
                pad=0.08
            )
            if scale =='log':
                cbar.locator = LogLocator(base=10.0,subs=[1,2,5], numticks=10)
                cbar.formatter = LogFormatter(base=10, labelOnlyBase=False) 
                cbar.update_ticks()
            cbar.set_label('Mixed Layer Depth (m)', fontsize=12)
        
        plot_name = f"MLD_winter_summer_{self.dates[-1].year}.pdf"
        plt.savefig(self.folder_path / plot_name,
                    dpi=200, bbox_inches='tight')
        plt.close()
    
        print('MLD plot done')

    def velocity(self, n_years=10):
        idx = self.get_time_indices(n_years)
        u = np.mean(
            self.ds.variables["u"][idx, -1, :, :],
            axis=0
        )
        v = np.mean(
            self.ds.variables["v"][idx, -1, :, :],
            axis=0
        )
        u = u[:, self.sort_idx]
        v = v[:, self.sort_idx]
        speed = np.sqrt(u**2 + v**2)
        year = self.dates[idx[-1]].year
        cmap = cmr.ocean
        self.plot_map(
            self.xt, self.yt, speed,
            f"Surface velocity (last {n_years} years)",
            f"Velocity_{year}.pdf",
            cmap=cmap,
            u=u, v=v
        )
        print("Velocity done")

    def energy(self):
        eke_m = self.ds_energy['k_m'][:]

        x = [datetime(d.year, d.month, d.day) for d in self.nrj_dates]
        fig, ax = plt.subplots(figsize=(9, 4))
         
        ax.plot(x, eke_m,color='black', lw=0.5)
        ax.set_xlabel("Time")
        ax.set_ylabel(r"$k_m$")
        ax.set_title(r"$k_m$ convergence")
        
        locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
        formatter = mdates.ConciseDateFormatter(locator)
        
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(6))
        
        ax.grid(True,alpha=0.3)
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.savefig(self.folder_path/'Energy_plot.pdf')
        plt.close()
        print("Energy done")
    

    def run_all(self, n_years_default=10):
        self.sst(n_years_default,vert=True)
        self.sst(n_years_default,vert=False)
        self.sss(n_years_default,vert=True)
        self.sss(n_years_default,vert=False)
        self.ssh(n_years_default)
        self.velocity(n_years_default)
        self.energy()
        self.heat_flux(n_years_default)
        self.mld(n_years=1)

start = datetime(1986,1,1)
run_dir = "/Odyssey/private/e25cheve/simu_veros/runs/global_1deg_glorys"
D = Diagnostics(run_dir,start)
D.run_all()




