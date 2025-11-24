import pandas as pd
import numpy as np
import math
import unittest

# --- Constants & Configuration ---
GRAVITY = 9.81  # m/s^2
PREDICTION_HORIZON_T = 2.0  # seconds
BRAKING_THRESHOLD_G = -0.25
BRAKING_PRESSURE_THRESHOLD = 1.0  # bar
CORNERING_DECEL_ESTIMATE_MS2 = 5.0  # Conservative decel if cornering expected
MIN_CLOSING_SPEED = 0.1  # m/s
MAX_GAP_FOR_DLS = 2000.0  # meters

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees).
    Returns distance in meters.
    """
    R = 6371000  # Radius of Earth in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi / 2.0)**2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def sigmoid(x):
    """
    Standard sigmoid function 1 / (1 + e^-x).
    """
    # Clip x to avoid overflow
    x = max(-20, min(20, x))
    return 1.0 / (1.0 + math.exp(-x))

def get_float(data, key, default=0.0):
    """Safely get a float from a dict/Series."""
    try:
        val = data.get(key, default)
        if pd.isna(val):
            return default
        return float(val)
    except:
        return default

def calculate_overtake_metrics(hero, rival):
    """
    Calculates the overtake probability and related metrics.
    
    Args:
        hero (dict/Series): Telemetry for the ego vehicle.
        rival (dict/Series): Telemetry for the rival vehicle.
        
    Returns:
        dict: Contains probability, decision, scores, and debug info.
    """
    # 1. Unit Conversion
    # Speed: km/h -> m/s
    v_hero = get_float(hero, 'Speed') / 3.6
    v_rival = get_float(rival, 'Speed') / 3.6
    
    # Acceleration: G -> m/s^2
    ax_hero = get_float(hero, 'accx_can') * GRAVITY
    ax_rival = get_float(rival, 'accx_can') * GRAVITY
    ay_hero = get_float(hero, 'accy_can') * GRAVITY
    
    # 2. Short-term Prediction (T seconds)
    # v_future = max(0, v + a * T)
    v_future_hero = max(0.0, v_hero + ax_hero * PREDICTION_HORIZON_T)
    v_future_rival = max(0.0, v_rival + ax_rival * PREDICTION_HORIZON_T)
    
    # 3. Gap Calculation
    gap = float('inf')
    
    # Try Laptrigger_lapdist_dls first
    dls_hero = get_float(hero, 'Laptrigger_lapdist_dls', -1)
    dls_rival = get_float(rival, 'Laptrigger_lapdist_dls', -1)
    
    used_gps = False
    if dls_hero >= 0 and dls_rival >= 0:
        diff = abs(dls_hero - dls_rival)
        if diff < MAX_GAP_FOR_DLS:
            gap = diff
        else:
            # Fallback to GPS if DLS diff is huge (e.g. start/finish line wrap, though simple abs doesn't handle wrap)
            # Assuming linear track for this check as per requirements
            used_gps = True
    else:
        used_gps = True
        
    if used_gps or gap == float('inf'):
        lat1 = get_float(hero, 'VBOX_Lat_Min')
        lon1 = get_float(hero, 'VBOX_Long_Min')
        lat2 = get_float(rival, 'VBOX_Lat_Min')
        lon2 = get_float(rival, 'VBOX_Long_Min')
        
        # Check if GPS is valid (non-zero)
        if lat1 != 0 and lat2 != 0:
            gap = haversine_distance(lat1, lon1, lat2, lon2)
        else:
            # If GPS is missing, we can't compute gap reliably.
            # Fail gracefully with a large gap.
            gap = 100.0

    # 4. Closing Speed and Time to Close
    closing_speed = v_future_hero - v_future_rival
    
    if closing_speed > MIN_CLOSING_SPEED:
        time_to_close = gap / closing_speed
    else:
        time_to_close = float('inf')
        
    # 5. Estimate Time Before Braking/Cornering
    # Check if rival is braking
    rival_pbrake_f = get_float(rival, 'pbrake_f')
    rival_pbrake_r = get_float(rival, 'pbrake_r')
    rival_accx_g = get_float(rival, 'accx_can')
    
    is_braking = (rival_pbrake_f > BRAKING_PRESSURE_THRESHOLD or 
                  rival_pbrake_r > BRAKING_PRESSURE_THRESHOLD or 
                  rival_accx_g < BRAKING_THRESHOLD_G)
                  
    if is_braking:
        # Estimate decel from rival accx (magnitude)
        # Clamp to reasonable range [2, 15] m/s^2
        decel = abs(rival_accx_g * GRAVITY)
        decel = max(2.0, min(15.0, decel))
    else:
        # Conservative decel for upcoming corner
        decel = CORNERING_DECEL_ESTIMATE_MS2
        
    # Distance before brake (approximate stopping/slowing distance needed)
    # Formula: v^2 / (2 * a)
    dist_before_brake = (v_hero ** 2) / (2 * decel)
    
    # Time before brake
    # t = d / v
    time_before_brake = dist_before_brake / max(v_hero, 1.0) # Avoid div/0
    
    # 6. Lateral and Defending Heuristics
    # space_ok: ego lateral acc small, ego steer small, rival steer not large
    ego_steer = abs(get_float(hero, 'Steering_Angle'))
    rival_steer = abs(get_float(rival, 'Steering_Angle'))
    
    # Thresholds (heuristic)
    # Lateral Acc < 0.3 G (~3 m/s^2) means we are not mid-corner limit
    # Steer < 20 degrees means we are relatively straight
    space_ok = (abs(get_float(hero, 'accy_can')) < 0.3 and 
                ego_steer < 20.0 and 
                rival_steer < 30.0) # Rival not swerving wildly
                
    # 7. Scores
    # Speed Score: Sigmoid on closing speed
    # Target: 2 m/s -> 0.88, 0.5 m/s -> 0.62
    # Sigmoid(x) matches this well.
    speed_score = sigmoid(closing_speed)
    
    # Gap Score: 1 - clamp(time_to_close / time_before_brake)
    # If we have lots of time before braking compared to time to close, score is high.
    if time_before_brake > 0:
        ratio = time_to_close / time_before_brake
    else:
        ratio = 1.0 # If time_before_brake is 0 (stopped?), ratio is bad
        
    gap_score = 1.0 - max(0.0, min(1.0, ratio))
    
    # Space Score
    space_score = 1.0 if space_ok else 0.0
    
    # Combined Probability
    probability = (0.5 * speed_score) + (0.35 * gap_score) + (0.15 * space_score)
    
    # 8. Decision
    # True if prob >= 0.65 AND time_to_close < time_before_brake AND space_ok
    decision = (probability >= 0.65 and 
                time_to_close < time_before_brake and 
                space_ok)
                
    # 9. AI Engineer Feedback
    feedback = []
    
    # Speed Delta
    speed_diff = v_hero - v_rival
    if abs(speed_diff) > 2.0:
        if speed_diff > 0:
            feedback.append(f"You are {speed_diff*3.6:.0f} km/h FASTER")
        else:
            feedback.append(f"Rival is {abs(speed_diff)*3.6:.0f} km/h FASTER")
            
    # Braking
    if is_braking:
        if get_float(hero, 'pbrake_f') < 5.0:
             feedback.append("Rival is BRAKING earlier")
    
    # Throttle
    hero_aps = get_float(hero, 'aps')
    rival_aps = get_float(rival, 'aps')
    if rival_aps > hero_aps + 20:
        feedback.append("Rival is on THROTTLE earlier")
        
    return {
        "probability": probability,
        "decision": decision,
        "metrics": {
            "gap": gap,
            "closing_speed": closing_speed,
            "time_to_close": time_to_close,
            "time_before_brake": time_before_brake,
            "space_ok": space_ok,
            "speed_score": speed_score,
            "gap_score": gap_score
        },
        "feedback": feedback
    }

class RaceAnalysis:
    def __init__(self, dataframe):
        self.df = dataframe
        self.vehicle_ids = self.df['vehicle_id'].unique()
        
        # Optimization: Dictionary of DataFrames per vehicle
        self.vehicle_data = {vid: self.df[self.df['vehicle_id'] == vid] for vid in self.vehicle_ids}
            
    def get_fastest_lap(self):
        """
        Finds the vehicle_id and Lap_Number of the fastest lap.
        """
        # Fast aggregation
        # Filter out partial laps (assuming a lap is > 60 seconds)
        if 'Lap_Number' not in self.df.columns or 'timestamp' not in self.df.columns:
            return None, None
            
        lap_times = self.df.groupby(['vehicle_id', 'Lap_Number'])['timestamp'].agg(lambda x: x.max() - x.min())
        
        # Filter reasonable laps (e.g., > 60s and < 5 mins)
        valid_laps = lap_times[(lap_times.dt.total_seconds() > 60) & (lap_times.dt.total_seconds() < 300)]
        
        if valid_laps.empty:
            return None, None

        best_idx = valid_laps.idxmin() # Returns (vehicle_id, Lap_Number)
        return best_idx

    def get_car_state_at_distance(self, vehicle_id, target_distance, target_lap=None):
        if vehicle_id not in self.vehicle_data:
            return None
            
        v_df = self.vehicle_data[vehicle_id]
        
        # SAFETY CHECK
        if 'Lap_Number' not in v_df.columns:
            # print(f"Error: Lap_Number missing for {vehicle_id}")
            return None
            
        # 1. Filter by Lap
        if target_lap is not None:
            v_df = v_df[v_df['Lap_Number'] == target_lap]
        
        if v_df.empty:
            return None

        # 2. Find closest distance
        # Use searchsorted for speed if sorted, but idxmin is safe
        if 'Laptrigger_lapdist_dls' in v_df.columns:
            idx = (v_df['Laptrigger_lapdist_dls'] - target_distance).abs().idxmin()
            return v_df.loc[idx]
        return None

    def analyze_sectors(self, hero_df, rival_df):
        """
        Calculates sector times for Hero and Rival.
        Splits track into 3 equal distance sectors.
        """
        if hero_df.empty or rival_df.empty:
            return None
            
        max_dist = hero_df['Laptrigger_lapdist_dls'].max()
        sector_len = max_dist / 3.0
        
        sectors = []
        for i in range(3):
            start_dist = i * sector_len
            end_dist = (i + 1) * sector_len
            
            # Filter data for sector
            h_sec = hero_df[(hero_df['Laptrigger_lapdist_dls'] >= start_dist) & (hero_df['Laptrigger_lapdist_dls'] < end_dist)]
            r_sec = rival_df[(rival_df['Laptrigger_lapdist_dls'] >= start_dist) & (rival_df['Laptrigger_lapdist_dls'] < end_dist)]
            
            if not h_sec.empty and not r_sec.empty:
                h_time = h_sec['timestamp'].max() - h_sec['timestamp'].min()
                r_time = r_sec['timestamp'].max() - r_sec['timestamp'].min()
                
                # Handle potential NaT or errors
                try:
                    h_s = h_time.total_seconds()
                    r_s = r_time.total_seconds()
                    delta = h_s - r_s # Negative means Hero is faster
                    
                    sectors.append({
                        "sector": i+1,
                        "hero_time": h_s,
                        "rival_time": r_s,
                        "delta": delta
                    })
                except:
                    continue
                    
        return sectors

    def calculate_overtake_probability(self, hero_state, rival_state):
        """
        Wrapper for the new calculate_overtake_metrics function.
        Maintains compatibility with app.py (returns score 0-100, reasons list).
        """
        if hero_state is None or rival_state is None:
            return 0, ["Missing Data"]
            
        result = calculate_overtake_metrics(hero_state, rival_state)
        
        prob_percent = result['probability'] * 100
        metrics = result['metrics']
        decision = result['decision']
        
        reasons = []
        if decision:
            reasons.append("Overtake Feasible")
            
        if metrics['closing_speed'] > 0.5:
            reasons.append(f"High Closing Speed ({metrics['closing_speed']:.1f} m/s)")
        elif metrics['closing_speed'] > 0:
            reasons.append(f"Closing ({metrics['closing_speed']:.1f} m/s)")
        else:
            reasons.append("Not Closing")
            
        if metrics['space_ok']:
            reasons.append("Space Available")
        else:
            reasons.append("No Space / Cornering")
            
        if metrics['time_to_close'] < metrics['time_before_brake']:
            reasons.append("Time Window Open")
        else:
            reasons.append("Braking Zone Too Close")
            
        return prob_percent, reasons, metrics, result.get('feedback', [])

# --- Unit Tests ---
class TestOvertakeModel(unittest.TestCase):
    def test_overtake_feasible(self):
        # Scenario: Hero faster, close, straight road
        hero = {
            'Speed': 150.0, # km/h
            'accx_can': 0.1, # G
            'accy_can': 0.0,
            'Steering_Angle': 0.0,
            'Laptrigger_lapdist_dls': 1000.0,
            'VBOX_Lat_Min': 0.0, 'VBOX_Long_Min': 0.0
        }
        rival = {
            'Speed': 130.0, # km/h -> ~20km/h diff = 5.5 m/s
            'accx_can': 0.0,
            'accy_can': 0.0,
            'Steering_Angle': 0.0,
            'pbrake_f': 0.0, 'pbrake_r': 0.0,
            'Laptrigger_lapdist_dls': 1010.0, # 10m gap
            'VBOX_Lat_Min': 0.0, 'VBOX_Long_Min': 0.0001
        }
        
        result = calculate_overtake_metrics(hero, rival)
        
        # Closing speed ~5.5 m/s -> Speed score ~0.99
        # Gap 10m. Time to close ~1.8s.
        # Time before brake: v=41m/s. a=5. d=41^2/10=168m. t=168/41=4s.
        # Time to close (1.8) < Time before brake (4). Gap score ~ 1 - 1.8/4 = 0.55.
        # Space ok = 1.
        # Prob = 0.5*0.99 + 0.35*0.55 + 0.15*1 = 0.495 + 0.19 + 0.15 = 0.835
        
        self.assertTrue(result['decision'], "Decision should be True for easy overtake")
        self.assertGreater(result['probability'], 0.8)

    def test_tight_corner_ahead(self):
        # Scenario: Tight corner, rival braking hard, no space
        hero = {
            'Speed': 100.0,
            'accx_can': -0.5,
            'accy_can': 0.5, # Turning
            'Steering_Angle': 30.0, # Steering
            'Laptrigger_lapdist_dls': 1000.0,
            'VBOX_Lat_Min': 0.0, 'VBOX_Long_Min': 0.0
        }
        rival = {
            'Speed': 90.0,
            'accx_can': -1.0, # Braking hard
            'accy_can': 0.5,
            'Steering_Angle': 30.0,
            'pbrake_f': 50.0, 'pbrake_r': 50.0,
            'Laptrigger_lapdist_dls': 1050.0,
            'VBOX_Lat_Min': 0.0, 'VBOX_Long_Min': 0.0001
        }
        
        result = calculate_overtake_metrics(hero, rival)
        
        # Space ok should be false due to steering/accy
        self.assertFalse(result['metrics']['space_ok'])
        self.assertFalse(result['decision'])
        self.assertLess(result['probability'], 0.65)

if __name__ == "__main__":
    # Run tests
    # unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    # Example Usage
    print("--- Overtake Model Example ---")
    hero_example = {
        'Speed': 160.0, 'Gear': 4, 'nmot': 6000, 'ath': 100, 'aps': 100,
        'pbrake_f': 0, 'pbrake_r': 0, 'accx_can': 0.2, 'accy_can': 0.05,
        'Steering_Angle': 2.0, 'VBOX_Long_Min': -86.0, 'VBOX_Lat_Min': 39.0,
        'Laptrigger_lapdist_dls': 1500.0
    }
    rival_example = {
        'Speed': 155.0, 'Gear': 4, 'nmot': 5800, 'ath': 100, 'aps': 100,
        'pbrake_f': 0, 'pbrake_r': 0, 'accx_can': 0.1, 'accy_can': 0.05,
        'Steering_Angle': 1.5, 'VBOX_Long_Min': -86.0001, 'VBOX_Lat_Min': 39.0001,
        'Laptrigger_lapdist_dls': 1515.0
    }
    
    res = calculate_overtake_metrics(hero_example, rival_example)
    print(f"Probability: {res['probability']:.2f}")
    print(f"Decision: {res['decision']}")
    print("Metrics:", res['metrics'])
    
    # Run Unit Tests manually to show output
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOvertakeModel)
    unittest.TextTestRunner(verbosity=2).run(suite)