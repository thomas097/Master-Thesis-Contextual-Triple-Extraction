import json


if __name__ == '__main__':
    with open('noun_identities.json', 'r') as file:
        data = json.load(file)

    while True:
        in_ = input('>> ')
        if in_ in data:
            for pron, val in sorted(data[in_].items(), key=lambda x: -x[1]):
                print(pron, val)
        else:
            print('I don\'t know')