
def log_val_q(data, writer, step):
    table = f"""
        | Val Step | pred_q  | target_q  | q_diff |
        |----------|-----------|-----------|-----------|
        | 1    | {data['pred_q'][0]} | {data['target_q'][0]} | {round(data['q_diff'][0], 2)} |
        | 2    | {data['pred_q'][1]} | {data['target_q'][1]} | {round(data['q_diff'][1], 2)} |
        | 3    | {data['pred_q'][2]} | {data['target_q'][2]} | {round(data['q_diff'][2], 2)} |
        | 4    | {data['pred_q'][3]} | {data['target_q'][3]} | {round(data['q_diff'][3], 2)} |
        | 5    | {data['pred_q'][4]} | {data['target_q'][4]} | {round(data['q_diff'][4], 2)} |
        | 6    | {data['pred_q'][5]} | {data['target_q'][5]} | {round(data['q_diff'][5], 2)} |
        | 7    | {data['pred_q'][6]} | {data['target_q'][6]} | {round(data['q_diff'][6], 2)} |
        | 8    | {data['pred_q'][7]} | {data['target_q'][7]} | {round(data['q_diff'][7], 2)} |
        | 9    | {data['pred_q'][8]} | {data['target_q'][8]} | {round(data['q_diff'][8], 2)} |
        | 10    | {data['pred_q'][9]} | {data['target_q'][9]} | {round(data['q_diff'][9], 2)} |
        | 11    | {data['pred_q'][10]} | {data['target_q'][10]} | {round(data['q_diff'][10], 2)} |
        | 12    | {data['pred_q'][11]} | {data['target_q'][11]} | {round(data['q_diff'][11], 2)} |
        | 13    | {data['pred_q'][12]} | {data['target_q'][12]} | {round(data['q_diff'][12], 2)} |
        | 14    | {data['pred_q'][13]} | {data['target_q'][13]} | {round(data['q_diff'][13], 2)} |
        | 15    | {data['pred_q'][14]} | {data['target_q'][14]} | {round(data['q_diff'][14], 2)} |
        | 16    | {data['pred_q'][15]} | {data['target_q'][15]} | {round(data['q_diff'][15], 2)} |
    """
    table = '\n'.join(l.strip() for l in table.splitlines())
    writer.add_text("table", table, step)