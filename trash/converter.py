labels_c3 = [
			"apple",
			"aquarium_fish",
			"baby",
			"bear",
			"beaver",
			"bed",
			"bee",
			"beetle",
			"bicycle",
			"bottle",
			"bowl",
			"boy",
			"bridge",
			"bus",
			"butterfly",
			"camel",
			"can",
			"castle",
			"caterpillar",
			"cattle",
			"chair",
			"chimpanzee",
			"clock",
			"cloud",
			"cockroach",
			"couch",
			"crab",
			"crocodile",
			"cup",
			"dinosaur",
			"dolphin",
			"elephant",
			"flatfish",
			"forest",
			"fox",
			"girl",
			"hamster",
			"house",
			"kangaroo",
			"keyboard",
			"lamp",
			"lawn_mower",
			"leopard",
			"lion",
			"lizard",
			"lobster",
			"man",
			"maple",
			"motorcycle",
			"mountain",
			"mouse",
			"mushroom",
			"oak",
			"orange",
			"orchid",
			"otter",
			"palm",
			"pear",
			"pickup_truck",
			"pine",
			"plain",
			"plate",
			"poppy",
			"porcupine",
			"possum",
			"rabbit",
			"raccoon",
			"ray",
			"road",
			"rocket",
			"rose",
			"sea",
			"seal",
			"shark",
			"shrew",
			"skunk",
			"skyscraper",
			"snail",
			"snake",
			"spider",
			"squirrel",
			"streetcar",
			"sunflower",
			"sweet_pepper",
			"table",
			"tank",
			"telephone",
			"television",
			"tiger",
			"tractor",
			"train",
			"trout",
			"tulip",
			"turtle",
			"wardrobe",
			"whale",
			"willow",
			"wolf",
			"woman",
			"worm"
		]
		
def c3_to_c2(y):
	
	if labels_c3[y] == "beaver" or labels_c3[y] == "dolphin" or labels_c3[y] == "otter" or labels_c3[y] == "seal" or labels_c3[y] == "whale":
		return 0
		
	if labels_c3[y] == "aquarium_fish" or labels_c3[y] == "flatfish" or labels_c3[y] == "ray" or labels_c3[y] == "shark" or labels_c3[y] == "trout":
		return 1
		
	if labels_c3[y] == "orchid" or labels_c3[y] == "poppy" or labels_c3[y] == "rose" or labels_c3[y] == "sunflower" or labels_c3[y] == "tulip":
		return 2
		
	if labels_c3[y] == "bottle" or labels_c3[y] == "bowl" or labels_c3[y] == "can" or labels_c3[y] == "cup" or labels_c3[y] == "plate":
		return 3

	if labels_c3[y] == "apple" or labels_c3[y] == "mushroom" or labels_c3[y] == "orange" or labels_c3[y] == "pear" or labels_c3[y] == "sweet_pepper":
		return 4
		
	if labels_c3[y] == "clock" or labels_c3[y] == "keyboard" or labels_c3[y] == "lamp" or labels_c3[y] == "telephone" or labels_c3[y] == "television":
		return 5
		
	if labels_c3[y] == "bed" or labels_c3[y] == "chair" or labels_c3[y] == "couch" or labels_c3[y] == "table" or labels_c3[y] == "wardrobe":
		return 6
		
	if labels_c3[y] == "bee" or labels_c3[y] == "beetle" or labels_c3[y] == "butterfly" or labels_c3[y] == "caterpillar" or labels_c3[y] == "cockroach":
		return 7
		
	if labels_c3[y] == "bear" or labels_c3[y] == "leopard" or labels_c3[y] == "lion" or labels_c3[y] == "tiger" or labels_c3[y] == "wolf":
		return 8
		
	if labels_c3[y] == "bridge" or labels_c3[y] == "castle" or labels_c3[y] == "house" or labels_c3[y] == "road" or labels_c3[y] == "skyscraper":
		return 9
		
	if labels_c3[y] == "cloud" or labels_c3[y] == "forest" or labels_c3[y] == "mountain" or labels_c3[y] == "plain" or labels_c3[y] == "sea":
		return 10
		
	if labels_c3[y] == "camel" or labels_c3[y] == "cattle" or labels_c3[y] == "chimpanzee" or labels_c3[y] == "elephant" or labels_c3[y] == "kangaroo":
		return 11
		
	if labels_c3[y] == "fox" or labels_c3[y] == "porcupine" or labels_c3[y] == "possum" or labels_c3[y] == "raccoon" or labels_c3[y] == "skunk":
		return 12
		
	if labels_c3[y] == "crab" or labels_c3[y] == "lobster" or labels_c3[y] == "snail" or labels_c3[y] == "spider" or labels_c3[y] == "worm":
		return 13
		
	if labels_c3[y] == "baby" or labels_c3[y] == "boy" or labels_c3[y] == "girl" or labels_c3[y] == "man" or labels_c3[y] == "woman":
		return 14
		
	if labels_c3[y] == "crocodile" or labels_c3[y] == "dinosaur" or labels_c3[y] == "lizard" or labels_c3[y] == "snake" or labels_c3[y] == "turtle":
		return 15
		
	if labels_c3[y] == "hamster" or labels_c3[y] == "mouse" or labels_c3[y] == "rabbit" or labels_c3[y] == "shrew" or labels_c3[y] == "squirrel":
		return 16
		
	if labels_c3[y] == "maple" or labels_c3[y] == "oak" or labels_c3[y] == "palm" or labels_c3[y] == "pine" or labels_c3[y] == "willow":
		return 17
		
	if labels_c3[y] == "bicycle" or labels_c3[y] == "bus" or labels_c3[y] == "motorcycle" or labels_c3[y] == "pickup_truck" or labels_c3[y] == "train":
		return 18
		
	if labels_c3[y] == "lawn_mower" or labels_c3[y] == "rocket" or labels_c3[y] == "streetcar" or labels_c3[y] == "tank" or labels_c3[y] == "tractor":
		return 19
		
	raise Exception(f'Label {y} does not exist.')
	
	
l = []
for i in range(20):
	l.append([])
	
for i in range(100):
	idx = c3_to_c2(i)
	l[idx].append(i)
	
for i in range(20):
	print(f'case {i}:')
	print (f'\treturn {l[i]}')	
