import random

def generate_dataset(n_samples=50000):

    subjects = ["Yo","Tú","Él","Ella","Nosotros","Ellos"]

    verbs = {
        "tener":["tengo","tienes","tiene","tiene","tenemos","tienen"],
        "comer":["como","comes","come","come","comemos","comen"]
    }

    objects = ["perro","gato","casa","libro","pizza"]
    adjectives = ["grande","pequeño","rojo"]
    preps = ["en casa","en Madrid"]

    dataset = []

    for _ in range(n_samples):

        idx = random.randint(0,5)
        subject = subjects[idx]

        verb_key = random.choice(list(verbs.keys()))
        correct_verb = verbs[verb_key][idx]

        obj = random.choice(objects)
        adj = random.choice(adjectives)
        prep = random.choice(preps).split()

        correct = [subject, correct_verb, "un", obj, adj] + prep

        wrong_idx = random.randint(0,5)
        wrong_verb = verbs[verb_key][wrong_idx]

        incorrect = [subject, wrong_verb, "un", obj, adj] + prep

        dataset.append((incorrect, correct))

    return dataset


def augment(sentence):

    if random.random() < 0.3:
        sentence.insert(1, "no")

    return sentence