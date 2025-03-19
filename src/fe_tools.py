import pandas as pd
import numpy as np
import statsmodels.api as sm
from tqdm.auto import tqdm
from . import ratios


def join_data_by_league(m_df, w_df):
    df_ = pd.concat([m_df.assign(League = 'M'), w_df.assign(League = 'W')], 
                    axis = 0, ignore_index=True)
    return df_

def add_ratios(data):
    # Расчет метрик для победившей команды
    detail_df = data.copy()
    detail_df['T1_eFG'] = ratios.calculate_eFG(detail_df['T1_FGM'], detail_df['T1_FGA'], detail_df['T1_FGM3'])
    detail_df['T1_FTR'] = ratios.calculate_FTR(detail_df['T1_FTA'], detail_df['T1_FGA'])
    detail_df['T1_TO_rate'] = ratios.calculate_TOV_rate(detail_df['T1_TO'], detail_df['T1_FGA'], detail_df['T1_FTA'])
    detail_df['T1_AST_TO_ratio'] = ratios.calculate_AST_TO_ratio(detail_df['T1_Ast'], detail_df['T1_TO'])
    detail_df['T1_OR_rate'] = ratios.calculate_ORB_rate(detail_df['T1_OR'], detail_df['T2_DR'])
    detail_df['T1_TR_rate'] = ratios.calculate_TRB_rate(detail_df['T1_OR'], detail_df['T1_DR'], detail_df['T2_DR'], detail_df['T2_OR'])
    detail_df['T1_Pace'] = ratios.calculate_Pace(detail_df['T1_FGM'], detail_df['T1_FGA'], detail_df['T1_FTA'], detail_df['T1_OR'], 
                                                 detail_df['T1_TO'], detail_df['T2_DR'])
    # Расчет метрик для проигравшей команды
    detail_df['T2_eFG'] = ratios.calculate_eFG(detail_df['T2_FGM'], detail_df['T2_FGA'], detail_df['T2_FGM3'])
    detail_df['T2_FTR'] = ratios.calculate_FTR(detail_df['T2_FTA'], detail_df['T2_FGA'])
    detail_df['T2_TO_rate'] = ratios.calculate_TOV_rate(detail_df['T2_TO'], detail_df['T2_FGA'], detail_df['T2_FTA'])
    detail_df['T2_AST_TO_ratio'] = ratios.calculate_AST_TO_ratio(detail_df['T2_Ast'], detail_df['T2_TO'])
    detail_df['T2_OR_rate'] = ratios.calculate_ORB_rate(detail_df['T2_OR'], detail_df['T1_DR'])
    detail_df['T2_TR_rate'] = ratios.calculate_TRB_rate(detail_df['T2_OR'], detail_df['T2_DR'], detail_df['T1_DR'], detail_df['T1_OR'])
    detail_df['T2_Pace'] = ratios.calculate_Pace(detail_df['T2_FGM'], detail_df['T2_FGA'], detail_df['T2_FTA'], detail_df['T2_OR'],
                                                 detail_df['T2_TO'], detail_df['T1_DR'])
    detail_df['Pace'] = detail_df[['T1_Pace', 'T2_Pace']].mean(axis = 1)
    detail_df['T1_OER'] = detail_df.eval("T1_Score/Pace")
    detail_df['T1_DER'] = detail_df.eval("T2_Score/Pace")
    return detail_df

def detail_preprocess(data):
    lose_cols = ['LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']
    win_cols = ['WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']
    common_cols = ['League', 'Season', 'DayNum', 'NumOT', 'WLoc']
    win_renamer = {col: "T1_" + col[1:] for col in ['WTeamID', 'WScore'] + win_cols}
    win_renamer.update({col: "T2_" + col[1:] for col in ['LTeamID', 'LScore'] + lose_cols})
    cols = common_cols + sorted(list(win_renamer.values()), key = lambda x: tuple(x.split('_')[::-1]))
    lose_renamer = {col: "T1_" + col[1:] for col in ['LTeamID', 'LScore'] + lose_cols}
    lose_renamer.update({col: "T2_" + col[1:] for col in ['WTeamID', 'WScore'] + win_cols})
    win_df = data.rename(columns = win_renamer)
    lose_df = data.rename(columns = lose_renamer)
    lose_df.loc[lose_df['WLoc'] == 'H', 'WLoc'] = 'A'
    lose_df.loc[lose_df['WLoc'] == 'A', 'WLoc'] = 'H'
    df = pd.concat([win_df[cols], lose_df[cols]], axis = 0, ignore_index = True)
    df['WLoc'] = df['WLoc'].map({'A': -1, 'N': 0, 'H': 1})
    df['PointDiff'] = df['T1_Score'] - df['T2_Score']
    df['win'] = np.where(df['PointDiff']>0,1,0)
    df = add_ratios(df)
    return df

def regular_statistics(regular_data):
    funcs = ['mean']
    suffix_cols = ['FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']
    boxscore_cols = ['PointDiff']
    boxscore_cols.extend(['T1_' + col for col in suffix_cols])
    boxscore_cols.extend(['T2_' + col for col in suffix_cols])
    season_statistics = regular_data.groupby(["League", "Season", "T1_TeamID"])[boxscore_cols].agg(funcs).reset_index()
    season_statistics.columns = [''.join(col).strip() for col in season_statistics.columns.values]
    return season_statistics

def last_days_stats(regular_data, cond = None, statname = 'win_ratio', period = 14):
    start_day = 132-period
    # T1
    if cond is None:
        lastdays_stats_T1 = regular_data.query(f"DayNum>{start_day}").reset_index(drop=True)
    else:
        lastdays_stats_T1 = regular_data.query(f"DayNum>{start_day} and ({cond})").reset_index(drop=True)
    lastdays_stats_T1 = lastdays_stats_T1.groupby(['League', 'Season','T1_TeamID'])['win'].mean()\
        .reset_index(name=f'T1_{statname}_{period}d')
    # T2
    if cond is None:
        lastdays_stats_T2 = regular_data.query(f"DayNum>{start_day}").reset_index(drop=True)
    else:
        lastdays_stats_T2 = regular_data.query(f"DayNum>{start_day} and ({cond})").reset_index(drop=True)
    lastdays_stats_T2 = lastdays_stats_T2.groupby(['League', 'Season','T2_TeamID'])['win'].mean()\
        .reset_index(name=f'T2_{statname}_{period}d')
    return lastdays_stats_T1, lastdays_stats_T2

def season_effects(regular_data, seeds, target = 'PointDiff'):
    reg_season_cols = ['League', 'Season', 'WLoc', 'DayNum', 'T1_TeamID','T2_TeamID']
    regular_season_effects = regular_data[reg_season_cols + [target]].copy()
    regular_season_effects['T1_TeamID'] = regular_season_effects['T1_TeamID'].astype(str)
    regular_season_effects['T2_TeamID'] = regular_season_effects['T2_TeamID'].astype(str)
    march_madness = pd.merge(seeds[['League', 'Season','TeamID']],seeds[['League', 'Season','TeamID']],
                             on=['League', 'Season'])
    march_madness.columns = ['League', 'Season', 'T1_TeamID', 'T2_TeamID']
    march_madness.T1_TeamID = march_madness.T1_TeamID.astype(str)
    march_madness.T2_TeamID = march_madness.T2_TeamID.astype(str)
    regular_season_effects = pd.merge(regular_season_effects, march_madness, 
                                      on = ['League', 'Season','T1_TeamID','T2_TeamID'])
    return regular_season_effects

def team_quality(regular_season_effects, target = 'PointDiff'):
    ### Original
    # Changed to Point Difference & Gaussian from Win & Binomial
    """
    formula = 'win~-1+T1_TeamID+T2_TeamID'
    glm = sm.GLM.from_formula(formula=formula, 
                              data=regular_season_effects.loc[regular_season_effects.Season==season,:], 
                              family=sm.families.Binomial()).fit()
    """
    formula = f'I({target}) ~ 1 + C(T1_TeamID) + C(T2_TeamID) + WLoc'
    glm = sm.GLM.from_formula(formula=formula,
                          data=regular_season_effects, 
                          family=sm.families.Gaussian()).fit()
    season = regular_season_effects['Season'].iloc[0]
    quality = pd.DataFrame(glm.params).reset_index()
    quality.columns = ['TeamID','quality']
    quality['Season'] = season
    #quality['quality'] = np.exp(quality['quality'])
    quality = quality.loc[quality.TeamID.str.contains('T1_')].reset_index(drop=True)
    
    quality['TeamID'] = quality['TeamID'].apply(lambda x: x[15:19]).astype(int)
    #quality['TeamID'] = quality['TeamID'].apply(lambda x: x[10:14]).astype(int)
    return quality

def team_quality_multiseason(regular_season_effects, seasons = None, target = 'PointDiff'):
    if seasons is None:
        seasons = sorted(regular_season_effects['Season'].unique().tolist())
    glm_quality = []
    for season in tqdm(seasons):
        if season == 2020: continue
        df_ = regular_season_effects.query(f"Season == {season}").copy()
        season_quality = team_quality(df_, target)
        glm_quality.append(season_quality)
    glm_quality = pd.concat(glm_quality, axis = 0, ignore_index = True)
    return glm_quality

def massey_ranking(mmassey):
    top_systems = [
        '7OT', 'AP', 'ARG', 'BBT', 'BIH', 'BWE', 'COL', 'DCI', 'DES', 'DII', 'DOK', 'DOL', 'DUN', 
        'EBP', 'ESR', 'FAS', 'HAS', 'JJK', 'JNG', 'KPK', 'LOG', 'MAS', 'MB', 'MOR', 'NOL', 'PGH', 
        'POM', 'REW', 'RPI', 'RT', 'RTH', 'SMS', 'SPR', 'TRK', 'TRP', 'USA', 'WIL', 'WLK', 'WOL']
    norm_df = mmassey.copy()
    # norm_df['norm_rank'] = norm_df.groupby(['Season', 'SystemName', 'RankingDayNum'])['OrdinalRank'].rank(pct=True)*100
    pivot_df = norm_df[norm_df['SystemName'].isin(top_systems)].pivot(
        index = ['Season', 'RankingDayNum', 'TeamID'], columns = 'SystemName', values = 'OrdinalRank')
    other_ranks = norm_df[~norm_df['SystemName'].isin(top_systems)]\
        .groupby(['Season', 'RankingDayNum', 'TeamID'])['OrdinalRank'].mean().rename("OTHR").reset_index()
    system_cols = top_systems + ['OTHR']
    pivot_df = pd.merge(pivot_df.reset_index(), other_ranks, 
                        how='outer', on = ['Season', 'RankingDayNum', 'TeamID'])
    pivot_df = pivot_df.sort_values(["Season", "RankingDayNum", "TeamID"])
    pivot_df.loc[:,system_cols] = pivot_df.groupby(["Season", "TeamID"])[system_cols].ffill()
    pivot_df['system_nan_cnt'] = pivot_df.isna().sum(axis = 1)
    mean_rank = pivot_df[system_cols].mean(axis = 1)
    for col in system_cols:
        pivot_df[col] = pivot_df[col].fillna(mean_rank)
    return pivot_df, system_cols

def gen_features(tourney_detailed, regular_detailed, tourney_seeds, mmassey, is_train = True):
    # preprocess detailed data
    regular_data = detail_preprocess(regular_detailed)
    if is_train:
        tourney_data = detail_preprocess(tourney_detailed)
    else:
        tourney_data = tourney_detailed.copy()
    # seeds
    seeds = tourney_seeds.copy()
    seeds['seed'] = seeds['Seed'].apply(lambda x: int(x[1:3]))
    # add regular season aggregations to tourney data
    season_statistics = regular_statistics(regular_data)
    season_statistics_T1 = season_statistics.copy()
    season_statistics_T2 = season_statistics.copy()
    season_statistics_T1.columns = ["T1_" + x.replace("T1_","").replace("T2_","opp_") for x in list(season_statistics_T1.columns)]
    season_statistics_T2.columns = ["T2_" + x.replace("T1_","").replace("T2_","opp_") for x in list(season_statistics_T2.columns)]
    season_statistics_T1.rename(columns = {'T1_League': 'League', 'T1_Season': 'Season'}, inplace = True)
    season_statistics_T2.rename(columns = {'T2_League': 'League', 'T2_Season': 'Season'}, inplace = True)
    tourney_data = pd.merge(tourney_data, season_statistics_T1, 
                            on = ['League', 'Season', 'T1_TeamID'], how = 'left')
    tourney_data = pd.merge(tourney_data, season_statistics_T2, 
                            on = ['League', 'Season', 'T2_TeamID'], how = 'left')
    # add last 14 days stats
    last14days_stats_T1, last14days_stats_T2 = last_days_stats(regular_data)
    away_stats_T1, away_stats_T2 = last_days_stats(regular_data, "WLoc != 1", 'away_win_ratio', 132)
    tourney_data = pd.merge(tourney_data, last14days_stats_T1, 
                            on = ['League', 'Season', 'T1_TeamID'], how = 'left')
    tourney_data = pd.merge(tourney_data, last14days_stats_T2, 
                            on = ['League', 'Season', 'T2_TeamID'], how = 'left')
    tourney_data = pd.merge(tourney_data, away_stats_T1, 
                            on = ['League', 'Season', 'T1_TeamID'], how = 'left')
    tourney_data = pd.merge(tourney_data, away_stats_T2, 
                            on = ['League', 'Season', 'T2_TeamID'], how = 'left')
    # season effects
    seasons = sorted(tourney_data['Season'].unique().tolist())
    for target, quality_name in zip(['PointDiff', 'T1_OER', 'T1_DER'], ['quality', 'OffQ', 'DefQ']):
        regular_season_effects = season_effects(regular_data, seeds, target)
        glm_quality = team_quality_multiseason(regular_season_effects, seasons, target=target)
        glm_quality_T1 = glm_quality.copy()
        glm_quality_T2 = glm_quality.copy()
        glm_quality_T1.columns = ['T1_TeamID',f'T1_{quality_name}','Season']
        glm_quality_T2.columns = ['T2_TeamID',f'T2_{quality_name}','Season']
        tourney_data = pd.merge(tourney_data, glm_quality_T1, on = ['Season', 'T1_TeamID'], how = 'left')
        tourney_data = pd.merge(tourney_data, glm_quality_T2, on = ['Season', 'T2_TeamID'], how = 'left')
    # add seed info
    seeds_T1 = seeds[['Season','TeamID','seed']].copy()
    seeds_T2 = seeds[['Season','TeamID','seed']].copy()
    seeds_T1.columns = ['Season','T1_TeamID','T1_seed']
    seeds_T2.columns = ['Season','T2_TeamID','T2_seed']
    tourney_data = pd.merge(tourney_data, seeds_T1, on = ['Season', 'T1_TeamID'], how = 'left')
    tourney_data = pd.merge(tourney_data, seeds_T2, on = ['Season', 'T2_TeamID'], how = 'left')
    tourney_data["Seed_diff"] = tourney_data["T1_seed"] - tourney_data["T2_seed"]
    # massey rank (men only)
    ratings_df, system_cols = massey_ranking(mmassey)
    agg_cols = system_cols + ['system_nan_cnt']
    last_ratings = ratings_df.query("RankingDayNum > 118").groupby(['Season', 'TeamID'])[agg_cols].mean()
    mid_ratings = ratings_df.query("100 < RankingDayNum <= 118").groupby(['Season', 'TeamID'])[agg_cols].mean()
    t1_last_renamer = {col: f"T1_{col}_last" for col in agg_cols}; t1_last_renamer['TeamID'] = 'T1_TeamID'
    t2_last_renamer = {col: f"T2_{col}_last" for col in agg_cols}; t2_last_renamer['TeamID'] = 'T2_TeamID'
    # t1_mid_renamer = {col: f"T1_{col}_mid" for col in agg_cols}; t1_mid_renamer['TeamID'] = 'T1_TeamID'
    # t2_mid_renamer = {col: f"T2_{col}_mid" for col in agg_cols}; t2_mid_renamer['TeamID'] = 'T2_TeamID'
    last_ratings_T1 = last_ratings.reset_index().rename(columns=t1_last_renamer)
    last_ratings_T2 = last_ratings.reset_index().rename(columns=t2_last_renamer)
    # mid_ratings_T1 = mid_ratings.reset_index().rename(columns=t1_mid_renamer)
    # mid_ratings_T2 = mid_ratings.reset_index().rename(columns=t2_mid_renamer)
    tourney_data = pd.merge(tourney_data, last_ratings_T1, on = ['Season', 'T1_TeamID'], how = 'left')
    tourney_data = pd.merge(tourney_data, last_ratings_T2, on = ['Season', 'T2_TeamID'], how = 'left')
    # tourney_data = pd.merge(tourney_data, mid_ratings_T1, on = ['Season', 'T1_TeamID'], how = 'left')
    # tourney_data = pd.merge(tourney_data, mid_ratings_T2, on = ['Season', 'T2_TeamID'], how = 'left')
    return tourney_data

