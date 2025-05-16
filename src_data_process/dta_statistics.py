
def data_analysis(data_list):
    stats_all = {}
    for i in range(len(data_list)):
        df = data_norm_list[i][self.curve_name_no_depth]
        # 基本统计量计算
        stats_df = pd.DataFrame({
            'Average': df.mean(),
            'S2': df.var(ddof=1),
            'S': df.std(ddof=1),
            'Median': df.median(),
            'Peak': df.kurtosis(),
            'Skewness': df.skew(),
            'Min': df.min(),
            '25%': df.quantile(0.25),
            '75%': df.quantile(0.75),
            'Max': df.max()
        })
        stats_all[self.char_well[i]] = stats_df


    df = self.data_norm_all[self.curve_name_no_depth]
    stats_df = pd.DataFrame({
        'Average': df.mean(),
        'S2': df.var(ddof=1),
        'S': df.std(ddof=1),
        'Median': df.median(),
        'Peak': df.kurtosis(),
        'Skewness': df.skew(),
        'Min': df.min(),
        '25%': df.quantile(0.25),
        '75%': df.quantile(0.75),
        'Max': df.max()
    })
    stats_all['All'] = stats_df
    return stats_all