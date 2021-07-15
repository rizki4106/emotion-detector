from src.detector import DetectEmotion

alg = DetectEmotion(gui=True)
res = alg.predict("img/team2.jpeg", "image")

print(res)
