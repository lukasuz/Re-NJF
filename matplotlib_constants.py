from matplotlib.font_manager import FontProperties

WIDTH  = 7.24137546
LINE_WIDTH = 1.
R = "#D90479"
B = "#04B2D9"
G = "#67BF63"
COLORS = [R, B, G, "#F5A623", "#8B5CF6", "#EF4444"]

s1 = 8
s2 = 6
lb_title  = FontProperties(size=s1, weight="bold")
lb_label  = FontProperties(size=s1, weight="bold")
lb_tick   = FontProperties(size=s2)
lb_legend = FontProperties(size=s2)
get_font = lambda weight, size: FontProperties(weight=weight, size=size)
