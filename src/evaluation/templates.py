ENTITIES = {'POSSESSION': ['my job', 'my cats', 'my studies', 'my overly attached girlfriend', 'my 2 kids', 'my chores',
                           'my tattoos', 'my blond hair', 'my old life', 'my two cats', 'my house', 'my card collection',
                           'my nose', 'my family', 'my name', 'my hair', 'my new car', 'my home-grown vegetables',
                           'my favorite show', 'my new laptop', 'my mother in law', 'my hobbies', 'my comic book collection',
                           'my neighborhood', 'my master\'s degree', 'my mood', 'my business', 'my four cats',
                           'my alter-ego', 'my favorite book', 'my favorite artist', 'my allergies', 'my freckles',
                           'my workplace', 'my university', 'my hometown', 'a lot of allergies'],
            'IND_POSSESSION': ['a job', 'two cats', 'an overly attached girlfriend', '2 kids', 'chores to do',
                               'tattoos', 'blond hair', 'many cats', 'an appartment downtown', 'a postcard collection',
                               'a nose', 'a family', 'a name', 'hair', 'a new car', 'a knack for sports',
                               'no favorite show', 'a new laptop', 'a mother in law', 'no hobbies', 'a comic book collection',
                               'a quiet neighborhood', 'a master\'s degree', 'a bad mood', 'a business', 'four trophies',
                               'an alter-ego', 'a favorite book', 'a favorite artist', 'allergies', 'freckles',
                               'a private office', 'a lot of allergies'],
            'OBJECTS': ['crime novels', 'the color blue', 'high school', 'new things', 'tattoos', 'pizza', 'mexican food',
                        'gummy bears', 'that store', 'music', 'books', 'video games', 'heavy metal music', 'seafood',
                        'sports', 'a fight', 'paintings', 'musical instruments', 'game of thrones', 'airplanes', 'south park'],
            'OCCUPATIONS': ['teachers', 'truck drivers', 'telemarketers', 'lawyers', 'athletes', 'students', 'soldiers',
                            'doctors', 'nurses', 'fishermen', 'accountants', 'engineers', 'construction workers',
                            'dancers', 'artists', 'musicians', 'singers', 'waiters'],
            'PEOPLE': ['my friend Mark', 'my grandmother', 'my mom', 'my dad', 'my cousin', 'my two kids',
                       'my father', 'my mother', 'my uncle Jim', 'my little sister', 'my little brother', 'my girlfriend',
                       'my bigger brother', 'my boyfriend', 'the neighbors', 'Nina', 'his new friend', 'her housemate',
                       'Mark', 'Jimmy', 'Eliza', 'Michael', 'Thomas', 'Meghan', 'Lisa'],
            'ACTIVITY': ['skating', 'swimming', 'rollerblading', 'studying for my master\'s degree', 'reading comics',
                         'reading fantasy novels', 'having lunch', 'partying', 'watching old movies', 'drinking', 'dancing',
                         'playing music', 'reading lots of books', 'seeing friends', 'visiting family', 'going out of town',
                         'cooking', 'doing chores', 'going to university', 'being outdoors', 'sleeping in', 'going to parties',
                         'watering the plants'],
            'PLACES': ['Amsterdam', 'Vermont', 'the office', 'the university', 'the country i live in', 'the town i live in',
                       'the gas station', 'my house', 'the bar', 'school', 'new bars', 'a rural area', 'this small town up north',
                       'the downtown area', 'the lake', '12 national parks in the US'],
            'COUNTRY': ['the Netherlands', 'China', 'France', 'Italy', 'Spain', 'the US', 'America', 'Mexico', 'Belgium',
                        'Ukraine', 'Portugal', 'Japan', 'Korea', 'Sweden', 'the UK'],
            'SKILLS': ['football', 'math', 'kindness', 'grammar', 'yoga', 'peace of mind', 'dancing'],
            'CITY': ['New York', 'Berlin', 'Brussels', 'Paris', 'Washington', 'Milan', 'Lisbon', 'Tokyo', 'Los Angeles',
                     'the city of Amsterdam', 'the city of Barcelona', 'the city of Berline', 'a place called Philadelphia'],
            'HOBBIES': ['collecting books', 'collecting postcards', 'photography', 'painting', 'drawing', 'seeing my family',
                        'biking', 'seeing any sport', 'watching television', 'baking pies', 'travelling', 'bird watching',
                        'gardening', 'gambling', 'day drinking', 'sightseeing', 'making people happy', 'playing outside'],
            'PER_PROPERTY': ['kind', 'quiet', 'a fast runner', 'French', 'Italian', 'a fast eater', 'in a good mood'],
            'STATE': ['in love', 'diagnosed with OCD'],
            'TIME': ['in high school', 'at the doctor\'s office', 'in 1997'],
            'INSTRUMENT': ['the violin', 'guitar', 'piano', 'the piano', 'drums', 'in a band'],
            'SPORT': ['hockey', 'football', 'video games', 'ice hockey', 'baseball', 'basketball'],
            'COMPANY': ['a law firm', 'the bank', 'the university', 'my grand dad', 'my dad', 'a local business', 'a brewery']}


PERSONA_TEMPLATES = {'not_like': ['i do not like POSSESSION .',
                                  'i do not like OBJECTS .',
                                  'i do not like OCCUPATIONS .',
                                  'i do not like PEOPLE .',
                                  'i do not like ACTIVITY .',
                                  'i do not like PLACES .'],
                     'do_like': ['i like POSSESSION .',
                                 'i like OBJECTS .',
                                 'i like OCCUPATIONS .',
                                 'i like PEOPLE .',
                                 'i like ACTIVITY .',
                                 'i like PLACES .'],
                     'not_love': ['i don\'t love POSSESSION .',
                                  'i don\'t love OBJECTS .',
                                  'i don\'t love OCCUPATIONS .',
                                  'i don\'t love PEOPLE .',
                                  'i don\'t love ACTIVITY .',
                                  'i don\'t love PLACES .'],
                     'do_love': ['i love POSSESSION .',
                                 'i love OBJECTS .',
                                 'i love OCCUPATIONS .',
                                 'i love PEOPLE .',
                                 'i love ACTIVITY .',
                                 'i love PLACES .'],
                     'not_hate': ['i do not hate POSSESSION .',
                                  'i do not hate OBJECTS .',
                                  'i do not hate OCCUPATIONS .',
                                  'i do not hate PEOPLE .',
                                  'i do not hate ACTIVITY .',
                                  'i do not hate PLACES .'],
                     'do_hate': ['i hate POSSESSION .',
                                 'i hate OBJECTS .',
                                 'i hate OCCUPATIONS .',
                                 'i hate PEOPLE .',
                                 'i hate ACTIVITY .',
                                 'i hate PLACES .'],
                     'not_enjoy': ['i do not enjoy ACTIVITY .',
                                   'i do not enjoy HOBBIES .'],
                     'do_enjoy': ['i enjoy ACTIVITY .',
                                  'i enjoy HOBBIES .'],
                     'do_work_for': ['i work for COMPANY'],
                     'not_went_to': ['i did not go to PLACES .',
                                     'i have never been to PLACES .',
                                     'i have not visited COUNTRY .',
                                     'i have not visited CITY .'],
                     'went_to': ['i went to PLACES yesterday .',
                                 'i went to PLACES .',
                                 'i went to COUNTRY .',
                                 'i visited COUNTRY .',
                                 'i went to CITY .',
                                 'i visited CITY .',
                                 'i visited PLACES'],
                     'moved_to': ['i moved to COUNTRY last year .',
                                  'i moved to CITY last year .',
                                  'i moved to CITY .',
                                  'i moved to COUNTRY last month .',
                                  'i moved to CITY last month .',
                                  'i moved to COUNTRY .',
                                  'i moved to COUNTRY for PEOPLE .',
                                  'i moved to CITY with POSSESSION .',
                                  'i moved to COUNTRY with POSSESSION .'],
                     'do_live_in': ['i live in CITY .',
                                    'i live in COUNTRY .'],
                     'not_play': ['i can not play the INSTRUMENT .',
                                  'i do not play SPORT .'],
                     'do_play': ['i can play the INSTRUMENT .',
                                 'i play SPORT .',
                                 'i play SPORT in my spare time .',
                                 'i play INSTRUMENT in my spare time .'],
                     'not_have': ['i do not have IND_POSSESSION .',
                                  'i never had a IND_POSSESSION .'],
                     'do_have': ['i have IND_POSSESSION .',
                                 'i just got IND_POSSESSION .'],
                     'do_teach': ['i teach SKILLS .',
                                  'i educate PEOPLE on SKILLS .',
                                  'i teach PEOPLE .',
                                  'i educate PEOPLE .'],
                     'not_being': ['i am not PER_PROPERTY .'],
                     'do_being': ['i am PER_PROPERTY .',
                                  'i have always been PER_PROPERTY .'],
                     'born_in': ['i was born in CITY .',
                                 'i am from CITY .',
                                 'i am from COUNTRY .',
                                 'i was born in COUNTRY .'],
                     'hobbies': ['my hobbies are HOBBIES and HOBBIES .'],
                     'others_like': ['OCCUPATIONS like IND_POSSESSION .',
                                     'PEOPLE like IND_POSSESSION .',
                                     'OCCUPATIONS like PLACES .',
                                     'OCCUPATIONS like HOBBIES .'],
                     'not_others_like': ['OCCUPATIONS like IND_POSSESSION .',
                                         'PEOPLE do not like IND_POSSESSION .'],
                     'am_state': ['i was STATE TIME .',
                                  'i am STATE .',
                                  'i was STATE .',
                                  'i was STATE with POSSESSION .'],
                     }


PERSONA_QUESTIONS = {'not_like': ['do you like * ?',
                                  'what do you like ?'],
                     'do_like': ['do you like * ?',
                                 'what do you like ?'],
                     'not_love': ['do you love * ?',
                                  'what do you love ?'],
                     'do_love': ['do you love * ?',
                                 'what do you love ?'],
                     'not_hate': ['do you hate * ?',
                                  'what do you hate ?'],
                     'do_hate': ['do you hate * ?',
                                 'what do you hate ?'],
                     'not_enjoy': ['do you enjoy * ?',
                                   'what do you enjoy ?'],
                     'do_enjoy': ['do you enjoy * ?',
                                  'what do you enjoy ?'],
                     'do_work_for': ['do you work for * ?',
                                     'who do you work for ?',
                                     'what do you do for a living ?'],
                     'not_went_to': ['did you go to * ?',
                                     'have you been to * ?',
                                     'have you visited * ?'],
                     'went_to': ['did you go to * ?',
                                 'have you been to * ?',
                                 'have you visited * ?',
                                 'where did you go ?'],
                     'moved_to': ['did you moved to * ?',
                                  'where did you move to ?'],
                     'do_live_in': ['do you live in * ?',
                                    'do you still live in * ?',
                                    'where do you live ?'],
                     'not_play': ['do you play * ?',
                                  'what do you play?'],
                     'do_play': ['do you play * ?',
                                 'what do you play?'],
                     'not_have': ['do you have * ?',
                                  'have you ever had a * ?',
                                  'what do you have ?',
                                  'do you have anything ?'],
                     'do_have': ['do you have * ?',
                                 'have you ever had a * ?'],
                     'do_teach': ['do you teach * ?',
                                  'do you teach ?'],
                     'not_being': ['are you * ?'],
                     'do_being': ['are you * ?'],
                     'born_in': ['are you born in * ?',
                                 'are you from * ?',
                                 'where wer you born ?'],
                     'hobbies': ['is * your hobby ?',
                                 'what are your hobbies ?',
                                 'do you have a hobby ?'],
                     'others_like': ['do * like * ?',
                                     'what do * like ?'],
                     'not_others_like': ['do * like * ?',
                                         'what don\'t * like ?'],
                     'am_state': ['were you * ?',
                               'are you * ?'],
                     }