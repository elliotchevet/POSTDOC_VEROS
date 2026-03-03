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
import matplotlib.dates as mdates
from dateutil import tz
import gsw
import scienceplots
plt.style.use('science')

class Diagnostics:
    def __init__(self,run_dir,start_date):

        self.run_dir = Path(run_dir)
        self.folder_path = self.run_dir / "plots"
        self.folder_path.mkdir(exist_ok=True)

        self.ds = nc.Dataset(self.run_dir / "global_1deg.averages.nc")
        self.snapshot = nc.Dataset(self.run_dir / "global_1deg.snapshot.nc")

        self.xt = self.ds.variables["xt"][:]
        self.xt, self.sort_idx = self.cyclic(self.xt)
        self.xu = self.ds.variables['xu'][:]
        self.xu, self.sort_idx_u = self.cyclic(self.xu)
        self.yt = self.ds.variables["yt"][:]
        self.yu = self.ds.variables['yu'][:]
        self.zt = self.ds.variables["zt"][:]
        self.zw = self.ds.variables["zw"][:]

        time_var = self.ds.variables["Time"]
        
        self.dates = np.array([
            start_date + timedelta(days=float(t))
            for t in time_var[:]
        ])
        print(self.dates)

        time_var_snap = self.snapshot.variables["Time"]
         
        self.snapshot_dates = np.array([
            start_date + timedelta(days=float(t))
            for t in time_var_snap[:]
        ])

        self.epsilon = 1e-6

    def get_time_indices(self, n_years=10, use_snapshot=False):
        dates = self.snapshot_dates if use_snapshot else self.dates
        last_year = dates[-1].year
        first_year = last_year - n_years + 1
        idx = [i for i, d in enumerate(dates) if d.year >= first_year]
        return np.array(idx)

    def cyclic(self,x):
        x = ((x + 180) % 360) -180
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        return x,sort_idx

    def plot_map(self, lon, lat, field, title, filename,
                 cmap="coolwarm", levels=100,
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

        plt.colorbar(cf, fraction=0.046, pad=0.04)

        if u is not None and v is not None:
            ax.streamplot(
                lon, lat, u, v,
                color="k",
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

    def sst(self, n_years=10):

        idx = self.get_time_indices(n_years)

        sst = np.mean(
            self.ds.variables["temp"][idx, -1, :, :],
            axis=0
        )

        sst = sst[:, self.sort_idx]
        year = self.dates[idx[-1]].year

        self.plot_map(
            self.xt, self.yt, sst,
            f"SST (last {n_years} years)",
            f"SST_{year}.pdf"
        )

        print("SST done")

    def sss(self, n_years=10):

        idx = self.get_time_indices(n_years)

        sss = np.mean(
            self.ds.variables["salt"][idx, -1, :, :],
            axis=0
        )

        sss = sss[:, self.sort_idx]
        year = self.dates[idx[-1]].year

        self.plot_map(
            self.xt, self.yt, sss,
            f"SSS (last {n_years} years)",
            f"SSS_{year}.pdf"
        )

        print("SSS done")

    def ssh(self, n_years=10):

        idx = self.get_time_indices(n_years)

        ssh = np.mean(
            self.ds.variables["ssh"][idx, :, :],
            axis=0
        )

        ssh = ssh[:, self.sort_idx]
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

    def find_month(self,month):
        return np.array([i for i, d in enumerate(self.dates) if d.month == month])

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
    
        temp = self.ds.variables['temp'][idx, :, :, :]
        salt = self.ds.variables['salt'][idx, :, :, :]
    
        temp = temp[:, :, :, self.sort_idx]
        salt = salt[:, :, :, self.sort_idx]
    
        zt = self.zt
        lat = self.yt
    
        idx_winter = [i for i in idx if self.dates[i].month in [1, 2, 3, 4]]
        idx_summer = [i for i in idx if self.dates[i].month in [7, 8, 9, 10]]
    
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
    
        maxval = max(np.nanmax(mld_winter), np.nanmax(mld_summer))
        levels = np.arange(0, maxval, 25)
    
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
    
        for ax, field, title in zip(axs, fields, titles):
    
            cf = ax.contourf(
                self.xt, self.yt, field,
                levels=levels,
                cmap='viridis',
                extend='max',
                transform=ccrs.PlateCarree()
            )
    
            ax.add_feature(cfeature.GSHHSFeature('low', levels=[1, 2, 6]))
            ax.coastlines()
            ax.set_title(title, fontsize=16)
    
        cbar = fig.colorbar(
            cf,
            ax=axs,
            orientation='horizontal',
            fraction=0.06,
            pad=0.08
        )
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

        self.plot_map(
            self.xt, self.yt, speed,
            f"Surface velocity (last {n_years} years)",
            f"Velocity_{year}.pdf",
            u=u, v=v
        )

        print("Velocity done")

    def run_all(self, n_years_default=10):

        self.sst(n_years_default)
        self.sss(n_years_default)
        self.ssh(n_years_default)
        self.velocity(n_years_default)
#        self.heat_flux(n_years_default)
        self.mld(n_years=1)

start = datetime(1986,4,2)
run_dir = "/Odyssey/private/e25cheve/simu_veros/runs/global_1deg_glorys/output"
D = Diagnostics(run_dir,start)
#D.run_all()




