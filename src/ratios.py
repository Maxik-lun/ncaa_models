# Функции для расчета метрик
def calculate_eFG(FGM, FGA, FGM3):
    # Effective Field Goal Percentage
    return (FGM + 0.5 * FGM3) / FGA

def calculate_TS(Points, FGA, FTA):
    # True Shooting Percentage
    return Points / (2 * (FGA + 0.44 * FTA))

def calculate_FTR(FTA, FGA):
    # Free Throw Rate
    return FTA / FGA

def calculate_DRB_rate(DRB, Opp_ORB):
    # Defensive Rebounds/Opponent Missed Shots
    return DRB / (DRB + Opp_ORB)

def calculate_BLK_rate(Blk, Opp_FGA, Opp_FGA3):
    return Blk/(Opp_FGA - Opp_FGA3)

def calculate_TOV_rate(TOV, FGA, FTA):
    return TOV / (FGA + 0.44 * FTA + TOV)

def calculate_AST_TO_ratio(AST, TOV):
    return AST / TOV

def calculate_ORB_rate(ORB, Opp_DRB):
    return ORB / (ORB + Opp_DRB)

def calculate_TRB_rate(ORB, DRB, Opp_DRB, Opp_ORB):
    return (ORB + DRB)/(ORB + DRB + Opp_DRB + Opp_ORB)

def calculate_Pace(FGM, FGA, FTA, ORB, TOV, Opp_DRB, minutes=40):
    possessions = FGA + 0.44 * FTA - 1.07*ORB/(ORB + Opp_DRB)*(FGA - FGM) + TOV
    return (possessions / minutes) * 40

def calculate_3PA_rate(FGA3, FGA):
    return FGA3/FGA

def calculate_FTA_rate(FTA, FGA):
    return FTA/FGA

# Stl_rate = Stl/Poss
# Foul_rate = PF/Poss
