I worked at a company on a project titled "Context Specific Tagging of Multilingual Queries". I had to perform IOB tagging of a query into action and entity. For example "How to 
turn of notifications on iPhone 10", would be tagged as How- B-action, to- I-action, of-I-action, notifications- B-entity, on-O(ignore), iPhone as B-entity and 10 as I-entity. 
The queries were in 4 languages English, German, French and Chinese. When I joined, the existing model for tagging queries was a string matching based algorithm.There was a 
dictionary of words where each word was given a tag of action and entity. For example:

Word	Tag
Zoom	B-entity
turn	B-action
off	I-action
Bing	B-action

The problem with the existing model was:
1> Lack of context for example "Zoom app" , where Zoom would be B-entity, but "Zoom on this picture.", where zoom would be B-action.
2> Lack of multilinguality and hence not extensible to other languages apart from English.
3> Huge size due to constant increase in size of dictionary when extended to other languages.

I had the task of building a model for 4 languages with size of model altogethet >1GB.

I first decided to implement a CRF model to introduce the contextual mapping using a simple One Hot Encoding. I performed an experiment with 160k english queries in the training set, 50k 
english queries in the test set and a validation split of 0.8.
Conditional Random Field is a special case of Markov Random field wherein the graph satisfies the property : “When we condition the graph on X globally i.e. when the values of random 
variables in X is fixed or given, all the random variables in set Y follow the Markov property p(Yᵤ/X,Yᵥ, u≠v) = p(Yᵤ/X,Yₓ, Yᵤ~Yₓ), where Yᵤ~Yₓ signifies that Yᵤ and Yₓ are 
neighbors in the graph.” Thus the advantages of this model was that it would provide unilateral context and some degree of transfer learning using word embeddings.
As a result the size of the dataset (predefined dictionary) would reduce as the model would be able to predict new words without being fed each word.

But even with this model the dataset size was considerable for 4 languages because as I used One Hot Encoding, I did not use any predefined encoding and tokenisation resulting 
in huge liguistic modelling being required.

Then I decided to implement CRF with an Elmo embedding like BERT. BERT provides bidirectional context matching, comes with an inbuilt tokeniser and is pretrained on the Wikipedia 
corpus. Hence it could solve the still existing problem of insufficient context matching, I could get pre trained tokenisation and encodings thus introducing a huge degre  of transfer 
learning. Also for a non-Latin script like Chinese an approach like a plain one hot encoding was an impossibility because the tokenisation for a non-Latin script is different from
that of a Latin script and tokenisation is absoluteky necessary. I first thought of using a predefined tokeniser like pynlpir with CRF+OHE. But this caused a problem as the queries
were tagged with different tokenisation than the one produced by puynlpir and hence I could not create my dataset. Hence I decided to use BERT which gave universal character based t
tokenisation removing the above problem.

The final results I obtained were as follows:

Language | Query wise accuracy | Token wise accuracy | Token Wise Recall | F1 Score | Precision
English  | 95.5%               | 98.8%               |  0.98             | 98.1%    | 0.98
French   | 70.8%               | 91.2%               |  0.86             | 91.6%    | 0.91
German   | 71.5%               | 90.4%               |  0.88             | 90.5%    | 0.89
Chinese  | 60%                 | 87%                 |  0.86             | 84%      | 0.88

Size of model: 700MB

My model achieved a token wise accuracy of ~99% for english, ~92% for French and German and ~90% for Chinese, had a total pickle model size of 700MB and could be effectively 
deployed on the Query Annotation System by the team.
The following examples will illustrate the problem statement better:
English:
Query : "Turn off notiications."
As it is in english it does not need to be translated.
Dataset creation with string matching gives:
Turn : B-action, off-I-action, notifications: B-entity
The dataset is fed to the CRF+BERT model and the model is tested on separate set of queries and attains the accuracy above.
The model performs better than the existing algo on 90% of the queries tagging words which the algo misses.

Non-English
Query: "Tour d'abonnement." (French)
English translation:
"Turn of subscriptions."
Mapping from englidh to french: 0:3->0:3, 5:6->5:5, 8:20->8:17
Dataset creation with string matching gives:
Turn : B-action, off-I-action, subscriptions: B-entity
The corresponding french query gets tagged as "Tour:B-action, d->I-action, abonnement->B-entity" by using the mappings.
The dataset is fed to the CRF+BERT model and the model is tested on separate set of queries and attains the accuracy above.
The model performs better than the existing algo on 80% of the queries tagging words which the algo misses.

A full flowchart for english queries would be:
1> Get query
2> Get tags from dictionary using existing algo
3> Feed tagged trainset to model
4> Perform training
5> Test model on test set.
6> Perform error analysis on test queries to find out if the model overperforms the existing hard coded dictionary tags and provides contextual tags. (model overperforms in >90% cases)

A full flowchart for non-english queries in Latin script would be:
1> Get query
2> Get english translation of query with mapping
3> Remove all queries with mapping, transaltion or lingual errors. (about 50%)
3> Get tags from dictionary for english transaltion using existing algo
4> Map back to tag original query
5> Feed tagged trainset to model
6> Perform training
7> Test model on test set.
8> Perform error analysis on test queries to find out if the model overperforms the existing hard coded dictionary tags and provides contextual tags. (model overperforms in >87% cases)

A full flowchart for non-english queries in non-atin script would be:
1> Get query
2> Get english translation of query with mapping
3> Remove all queries with mapping, transaltion or lingual errors. (about 60%)
3> Get tags from dictionary for english transaltion using existing algo
4> Map back to tag original query
5> Tokenise the query to tag each individual character.
关闭 订阅 -> turn subscriptions
关闭 -> turn (B-action)
订阅 -> subscriptions (B-entity)
is tokenised as:
关 -> B-action
闭 -> I-action
订 -> B-entity
阅 -> I-entity
6> Feed tagged trainset to model
7> Perform training
8> Test model on test set.
9> Perform error analysis on test queries to find out if the model overperforms the existing hard coded dictionary tags and provides contextual tags. (model overperforms in >87% cases)

problems Faced:
1> Huge percentage of mapping errors including no mapping, multiple mapping
2> lingual differences in non-latin scripts

challenges:
1> Non-latin script
2> Size constraint
3> Accuracy threshold
4> Huge size of daatset
5> Transfer learning
