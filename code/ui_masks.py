import cv2
import numpy as np
import os



#Kill feed (white background)
kill_self_lower = np.array([0, 0, 230]) # High value (brightness) requirement
kill_self_upper = np.array([180, 25, 255]) # Low saturation (pure white)


#Kill feed (blue half = teamate killer)
kill_team_lower = np.array([86,  51,  14])
kill_team_upper = np.array([129, 255, 249])


#Kill feed (yellow half = enemy killer)
kill_enemy_lower = np.array([16,  51,  4])
kill_enemy_upper = np.array([44,  255, 253])

#Domination bar
dom_enemy_lower = np.array([35,  150, 150])
dom_enemy_upper = np.array([75,  255, 255])

# Domination bar — win indicator diamonds glow yellow when a point is won
dom_win_lower    = np.array([20,  150, 180])
dom_win_upper    = np.array([35,  255, 255])

# Health bar — white fill
health_lower     = np.array([0,   0,   180])
health_upper     = np.array([179, 40,  255])

# Overshield — bright cyan/blue segment on right of health bar
shield_lower     = np.array([90,  150, 150])
shield_upper     = np.array([110, 255, 255])

# Ult charge — blue/purple box (not ready)
ult_blue_lower   = np.array([110, 60,  60])
ult_blue_upper   = np.array([140, 255, 200])

# Ult charge — bright yellow flash  
ult_ready_lower  = np.array([20,  150, 200])
ult_ready_upper  = np.array([35,  255, 255])

# Ability cooldown underline — bright cyan when available
ability_lower    = np.array([90,  150, 150])
ability_upper    = np.array([110, 255, 255])

#Domination bar percentage (team)
dom_team_lower = np.array([100, 80,  80])
dom_team_upper = np.array([130, 255, 200])

#Domination bar — team indicator
dom_team_win_lower = np.array([100, 150, 180])
dom_team_win_upper = np.array([130, 255, 255])

#--------------

def apply_hsv_mask(bgr_image, lower, upper):
    """
    Convert a BGR crop to HSV and apply a color threshold mask.
    Returns the binary mask (255=match, 0=no match).
    
    This is the core CV operation — we convert color spaces then
    threshold each pixel based on whether it falls within our
    defined HSV range.
    """
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    return mask


def get_fill_ratio(mask):
    """
    Calculate what fraction of the mask is non-zero (matched pixels).
    Used to estimate how full a bar is as a 0.0-1.0 ratio.
    
    Math: count white pixels / total pixels in mask
    """
    total = mask.shape[0] * mask.shape[1]
    if total == 0:
        return 0.0
    return np.count_nonzero(mask) / total


def get_horizontal_fill(mask):
    """
    Estimate bar fill percentage by measuring how far the matched
    pixels extend horizontally across the image.
    
    Collapses the mask vertically (OR across rows) then finds the
    rightmost matched column. Useful for bars that fill left to right.
    
    Returns a 0.0-1.0 ratio of how far across the bar is filled.
    """
    # Collapse rows — any row having a match counts for that column
    col_sums = np.sum(mask, axis=0)
    matched_cols = np.where(col_sums > 0)[0]
    
    if len(matched_cols) == 0:
        return 0.0
    
    rightmost = matched_cols[-1]
    return rightmost / mask.shape[1]


def analyze_kill_feed(kill_feed_crop):
    white_mask  = apply_hsv_mask(kill_feed_crop, kill_self_lower,  kill_self_upper) 
    blue_mask   = apply_hsv_mask(kill_feed_crop, kill_team_lower,  kill_team_upper)
    yellow_mask = apply_hsv_mask(kill_feed_crop, kill_enemy_lower, kill_enemy_upper)

        # Check each row for white pixel coverage
    row_white = np.sum(white_mask, axis=1) / white_mask.shape[1]
    
    # A real kill entry row has VERY high white coverage (>50%)
    # Diagonal lines/explosions have scattered white pixels not full rows
    strong_white_rows = np.sum(row_white > 0.35)
    white_ratio = strong_white_rows / white_mask.shape[0]

    # Additional structural check:
    # Count how many CONSECUTIVE rows are white
    # Real kill entries are solid rectangles — at least 15 consecutive white rows
    max_consecutive = 0
    current_consecutive = 0
    for val in row_white:
        if val > 0.35:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    
    # Must have at least 15 consecutive white rows to be a real kill entry
    has_white_band = max_consecutive >= 10

    blue_ratio   = get_fill_ratio(blue_mask)
    yellow_ratio = get_fill_ratio(yellow_mask)

    self_kill  = has_white_band and white_ratio > 0.25
    team_kill  = has_white_band and blue_ratio > 0.05 and yellow_ratio > 0.02
    enemy_kill = has_white_band and yellow_ratio > 0.02 and blue_ratio < 0.03

    return {
        "self_kill":          self_kill,
        "team_kill":          team_kill,
        "enemy_kill":         enemy_kill,
        "white_ratio":        round(white_ratio, 3),
        "blue_ratio":         round(blue_ratio,  3),
        "yellow_ratio":       round(yellow_ratio, 3),
        "max_consec_white":   max_consecutive,
    }



def analyze_domination(point_pct_crop):
    enemy_mask    = apply_hsv_mask(point_pct_crop, dom_enemy_lower,    dom_enemy_upper)
    team_mask     = apply_hsv_mask(point_pct_crop, dom_team_lower,     dom_team_upper)
    win_mask      = apply_hsv_mask(point_pct_crop, dom_win_lower,      dom_win_upper)
    team_win_mask = apply_hsv_mask(point_pct_crop, dom_team_win_lower, dom_team_win_upper)

    enemy_fill     = get_horizontal_fill(enemy_mask)
    team_fill      = get_horizontal_fill(team_mask)
    enemy_won_point = get_fill_ratio(win_mask)      > 0.02
    team_won_point  = get_fill_ratio(team_win_mask) > 0.02

    return {
        "enemy_fill":      round(enemy_fill, 3),
        "team_fill":       round(team_fill, 3),
        "enemy_won_point": enemy_won_point,
        "team_won_point":  team_won_point,
    }


def analyze_health(health_bar_crop):
    """
    Analyze the health bar crop to estimate current health percentage
    and detect if an overshield is active.
    
    Health % = how far the white bar extends horizontally.
    Overshield = bright cyan pixels present on right side of bar.
    
    Returns a dict with:
    - health_pct: 0.0-1.0 estimated health ratio
    - overshield: True if overshield is active
    """
    health_mask = apply_hsv_mask(health_bar_crop, health_lower, health_upper)
    shield_mask = apply_hsv_mask(health_bar_crop, shield_lower, shield_upper)

    health_pct  = get_horizontal_fill(health_mask)
    overshield  = get_fill_ratio(shield_mask) > 0.02

    return {
        "health_pct": round(health_pct, 3),
        "overshield": overshield,
    }


def analyze_ult(ult_crop):
    """
    Ult charge box: blue = charging, yellow = ready (100%)
    
    Ability cooldown boxes:
    - Cyan fill PRESENT = ability is ON COOLDOWN (recharging)
    - Cyan fill ABSENT = ability is AVAILABLE (icon showing)
    """
    yellow_mask  = apply_hsv_mask(ult_crop, ult_ready_lower, ult_ready_upper)
    blue_mask    = apply_hsv_mask(ult_crop, ult_blue_lower,  ult_blue_upper)
    ability_mask = apply_hsv_mask(ult_crop, ability_lower,   ability_upper)

    ult_ready        = get_fill_ratio(yellow_mask)  > 0.05
    ult_active       = get_fill_ratio(blue_mask)    > 0.03
    ability_on_cooldown = get_fill_ratio(ability_mask) > 0.02  # cyan

    return {
        "ult_ready":            ult_ready,
        "ult_active":           ult_active,
        "ability_on_cooldown":  ability_on_cooldown,  #if true on cooldown, else available
    }





