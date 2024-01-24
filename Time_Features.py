import numpy as np
import pandas as pd


class TimeCovariates():
    """
    Extract all time covariates except for holidays.

    -moh: Minutes of hour
    -hod: Hour of day
    -dow: Day of week
    -dom: Day of month
    -doy: Day of year
    -moy: Month of year
    -woy: Week of year

    -Normalized: normalize data into the given range.
    -Normalize_range: the mapped range the data will be normalized into.
    """

    def __init__(
            self,
            datetimes,
            add_moh: bool = False,
            add_hod: bool = True,
            add_dow: bool = True,
            add_dom: bool = False,
            add_doy: bool = False,
            add_moy: bool = True,
            add_woy: bool = False,
            normalized=True,
            normalize_range=(0, 1)):

        self.dti = datetimes
        self.add_moh = add_moh
        self.add_hod = add_hod
        self.add_dow = add_dow
        self.add_dom = add_dom
        self.add_doy = add_doy
        self.add_moy = add_moy
        self.add_woy = add_woy
        self.normalized = normalized
        self.normalize_range = normalize_range

    def _minute_of_hour(self):
        minutes = np.array(self.dti.minute, dtype=np.float32)
        if self.normalized:
            minutes = minutes / 59.0 - self.normalize_range[0]
        return minutes

    def _hour_of_day(self):
        hours = np.array(self.dti.hour, dtype=np.float32)
        if self.normalized:
            hours = hours / 23.0 - self.normalize_range[0]
        return hours

    def _day_of_week(self):
        day_week = np.array(self.dti.dayofweek, dtype=np.float32)
        if self.normalized:
            day_week = day_week / 6.0 - self.normalize_range[0]
        return day_week

    def _day_of_month(self):
        day_month = np.array(self.dti.day, dtype=np.float32)
        if self.normalized:
            day_month = day_month / 30.0 - self.normalize_range[0]
        return day_month

    def _day_of_year(self):
        day_year = np.array(self.dti.dayofyear, dtype=np.float32)
        if self.normalized:
            day_year = day_year / 364.0 - self.normalize_range[0]
        return day_year

    def _month_of_year(self):
        month_year = np.array(self.dti.month, dtype=np.float32)
        if self.normalized:
            month_year = month_year / 11.0 - self.normalize_range[0]
        return month_year

    def _week_of_year(self):
        week_year = np.array(self.dti.strftime("%U").astype(int), dtype=np.float32)
        if self.normalized:
            week_year = week_year / 51.0 - self.normalize_range[0]
        return week_year

    def get_covariates(self):
        """Get all time covariates."""
        all_covs = []
        columns = []
        if self.add_moh:
            moh = self._minute_of_hour().reshape(1, -1)
            all_covs.append(moh)
            columns.append("moh")
        if self.add_hod:
            hod = self._hour_of_day().reshape(1, -1)
            all_covs.append(hod)
            columns.append("hod")
        if self.add_dow:
            dow = self._day_of_week().reshape(1, -1)
            all_covs.append(dow)
            columns.append("dow")
        if self.add_dom:
            dom = self._day_of_month().reshape(1, -1)
            all_covs.append(dom)
            columns.append("dom")
        if self.add_doy:
            doy = self._day_of_year().reshape(1, -1)
            all_covs.append(doy)
            columns.append("doy")
        if self.add_moy:
            moy = self._month_of_year().reshape(1, -1)
            all_covs.append(moy)
            columns.append("moy")
        if self.add_woy:
            woy = self._week_of_year().reshape(1, -1)
            all_covs.append(woy)
            columns.append("woy")

        covariates = pd.DataFrame(
            data=np.vstack(all_covs).transpose(),
            columns=columns,
            index=self.dti,
        )

        return covariates

    def get_feature_names(self):

        """Get all time covariates."""

        columns = []
        if self.add_moh:
            columns.append("moh")
        if self.add_hod:
            columns.append("hod")
        if self.add_dow:
            columns.append("dow")
        if self.add_dom:
            columns.append("dom")
        if self.add_doy:
            columns.append("doy")
        if self.add_moy:
            columns.append("moy")
        if self.add_woy:
            columns.append("woy")

        return columns
