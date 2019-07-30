package vader

import (
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"regexp"
	"strconv"
	"strings"

	"github.com/gonum/floats"
)

const (
	//(empirically derived mean sentiment intensity rating increase for booster words)
	bIncr = 0.293
	bDecr = -0.293

	//(empirically derived mean sentiment intensity rating increase for using ALLCAPs to emphasize a word)
	cIncr   = 0.733
	nScalar = -0.74

	lexiconFile      = "./vader_lexicon.txt"
	emojiLexiconFile = "./emoji_utf8_lexicon.txt"

	alpha     = 15   //constant for normalize
	includeNt = true //flag to check "n't" in negated
)

// RegexRemovePunctuation for removing punctuation
var RegexRemovePunctuation = regexp.MustCompile(fmt.Sprintf("[%s]", regexp.QuoteMeta(`!"//$%&'#()*+,-./:;<=>?@[\]^_{|}~`+"`")))

// Punctuations ..
var Punctuations = []string{".", "..", "...", "!", "?", ",", ";", ":", "-", "'", "\"", "!!", "!!!", "??", "???", "?!?", "!?!", "?!?!", "!?!?", "????", "?????"}

// Negations ..
var Negations = []string{"aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
	"ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
	"dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
	"don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
	"neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
	"oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
	"oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
	"without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"}

// BoosterMap booster/dampener 'intensifiers' or 'degree adverbs'
// http://en.wiktionary.org/wiki/Category:English_degree_adverbs
var BoosterMap = map[string]float64{"absolutely": bIncr, "amazingly": bIncr, "awfully": bIncr, "completely": bIncr, "considerably": bIncr,
	"decidedly": bIncr, "deeply": bIncr, "effing": bIncr, "enormously": bIncr,
	"entirely": bIncr, "especially": bIncr, "exceptionally": bIncr, "extremely": bIncr,
	"fabulously": bIncr, "flipping": bIncr, "flippin": bIncr,
	"fricking": bIncr, "frickin": bIncr, "frigging": bIncr, "friggin": bIncr, "fully": bIncr, "fucking": bIncr,
	"greatly": bIncr, "hella": bIncr, "highly": bIncr, "hugely": bIncr, "incredibly": bIncr,
	"intensely": bIncr, "majorly": bIncr, "more": bIncr, "most": bIncr, "particularly": bIncr,
	"purely": bIncr, "quite": bIncr, "really": bIncr, "remarkably": bIncr,
	"so": bIncr, "substantially": bIncr,
	"thoroughly": bIncr, "totally": bIncr, "tremendously": bIncr,
	"uber": bIncr, "unbelievably": bIncr, "unusually": bIncr, "utterly": bIncr,
	"very":   bIncr,
	"almost": bDecr, "barely": bDecr, "hardly": bDecr, "just enough": bDecr,
	"kind of": bDecr, "kinda": bDecr, "kindof": bDecr, "kind-of": bDecr,
	"less": bDecr, "little": bDecr, "marginally": bDecr, "occasionally": bDecr, "partly": bDecr,
	"scarcely": bDecr, "slightly": bDecr, "somewhat": bDecr,
	"sort of": bDecr, "sorta": bDecr, "sortof": bDecr, "sort-of": bDecr}

// SentimentLadenIdioms check for sentiment laden idioms that do not contain lexicon words (future work, not yet implemented)
var SentimentLadenIdioms = map[string]int{"cut the mustard": 2, "hand to mouth": -2,
	"back handed": -2, "blow smoke": -2, "blowing smoke": -2,
	"upper hand": 1, "break a leg": 2,
	"cooking with gas": 2, "in the black": 2, "in the red": -2,
	"on the ball": 2, "under the weather": -2}

// SpecialCaseIdioms check for special case idioms containing lexicon words
var SpecialCaseIdioms = map[string]float64{"the shit": 3, "the bomb": 3, "bad ass": 1.5, "yeah right": -2,
	"kiss of death": -1.5}

// Determine if input contains negation words
func negated(inputWords []string) bool {
	var inputWordsLowercased []string

	for _, inputWord := range inputWords {
		inputWordsLowercased = append(inputWordsLowercased, strings.ToLower(inputWord))
	}

	for i, word := range inputWordsLowercased {
		for _, negWord := range Negations {
			if negWord == word {
				return true
			}
		}

		if word == "least" {
			if i > 0 && inputWordsLowercased[i-1] != "at" {
				return true
			}
		}

		if includeNt {
			if strings.Contains(word, "n't") {
				return true
			}
		}
	}

	return false
}

// Normalize the score to be between -1 and 1 using an alpha that
// approximates the max expected value
func normalize(score float64) float64 {
	normalizedScore := score / math.Sqrt((score*score)+float64(alpha))

	if normalizedScore < -1.0 {
		return -1.0
	} else if normalizedScore > 1.0 {
		return 1.0
	} else {
		return normalizedScore
	}
}

//Check whether just some words in the input are ALL CAPS
//:param list words: The words to inspect
//:returns: `True` if some but not all items in `words` are ALL CAPS
func allcapDifferential(words []string) bool {
	var isDifferent bool
	var allcapWords int

	for _, word := range words {
		if word == strings.ToUpper(word) {
			allcapWords++
		}
	}

	capDifferential := len(words) - allcapWords
	if capDifferential > 0 && capDifferential < len(words) {
		isDifferent = true
	}

	return isDifferent
}

// Check if the preceding words increase, decrease, or negate/nullify the
// valence
func scalarIncDec(word string, valence float64, isCapDiff bool) float64 {
	var scalar float64

	wordLower := strings.ToLower(word)

	if value, ok := BoosterMap[wordLower]; ok {
		scalar = value
		if valence < 0 {
			scalar *= -1
		}
		//check if booster/dampener word is in ALLCAPS (while others aren't)
		if word == strings.ToUpper(word) && isCapDiff {
			if valence > 0 {
				scalar += cIncr
			} else {
				scalar -= cIncr
			}
		}
	}

	return scalar
}

// SentiText ..
type SentiText struct {
	Text              string
	WordsAndEmoticons []string
	IsCapDiff         bool
}

// Init ..
func (s *SentiText) Init(text string) {
	s.Text = text
	s.WordsAndEmoticons = s._wordsAndEmoticons()
	// doesn't separate words from
	// adjacent punctuation (keeps emoticons & contractions)
	s.IsCapDiff = allcapDifferential(s.WordsAndEmoticons)
}

//Returns mapping of form:
//{
//'cat,': 'cat',
//',cat': 'cat',
//}
func (s *SentiText) _wordsPlusPunc() map[string]string {
	noPuncText := RegexRemovePunctuation.ReplaceAllString(string(s.Text), "")

	//removes punctuation (but loses emoticons & contractions)
	wordsOnly := strings.Fields(noPuncText)

	// remove singletons
	var wordsOnlyWithoutSingletons []string
	for _, word := range wordsOnly {
		if len(word) > 1 {
			wordsOnlyWithoutSingletons = append(wordsOnlyWithoutSingletons, word)
		}
	}

	// the product gives ('cat', ',') and (',', 'cat')

	puncBefore := make(map[string]string)
	puncAfter := make(map[string]string)

	for _, p := range cartesianProduct(Punctuations, wordsOnly) {
		puncBefore[strings.Join(p, "")] = p[1]
	}

	for _, p := range cartesianProduct(wordsOnly, Punctuations) {
		puncAfter[strings.Join(p, "")] = p[0]
	}

	wordsPuncDict := puncBefore

	for key, value := range puncAfter {
		wordsPuncDict[key] = value
	}

	return wordsPuncDict
}

// Cartesian product of input iterables.
func cartesianProduct(arr1 []string, arr2 []string) [][]string {
	var result [][]string

	for _, item1 := range arr1 {
		for _, item2 := range arr2 {
			result = append(result, []string{item1, item2})
		}
	}

	return result
}

//Removes leading and trailing puncutation
//Leaves contractions and most emoticons
//Does not preserve punc-plus-letter emoticons (e.g. :D)
func (s *SentiText) _wordsAndEmoticons() []string {
	wes := strings.Fields(s.Text)
	wordsPuncDict := s._wordsPlusPunc()

	var wesCleaned []string
	for _, we := range wes {
		if len(we) > 1 {
			wesCleaned = append(wesCleaned, we)
		}
	}

	for i, we := range wesCleaned {
		if value, ok := wordsPuncDict[we]; ok {
			wesCleaned[i] = value
		}
	}

	return wesCleaned
}

// SentimentIntensityAnalyzer Give a sentiment intensity score to sentences.
type SentimentIntensityAnalyzer struct {
	LexiconMap        map[string]float64
	EmojiLexiconMap   map[string]string
	SpecialCaseIdioms map[string]float64
}

// Init sentiment analyzer with lexicons
// if no filepaths passed to init, using default lexicon files
func (sia *SentimentIntensityAnalyzer) Init(filenames ...string) error {
	var lexiconFilename string
	var emojiLexiconFilename string

	if len(filenames) == 2 {
		lexiconFilename = filenames[0]
		emojiLexiconFilename = filenames[1]
	} else {
		lexiconFilename = lexiconFile
		emojiLexiconFilename = emojiLexiconFile
	}

	// load lexicon file
	lexicon, err := ioutil.ReadFile(lexiconFilename)
	if err != nil {
		return err
	}

	sia.LexiconMap = sia.makeLexiconMap(string(lexicon))

	// load emoji lexicon file
	emojiLexicon, err := ioutil.ReadFile(emojiLexiconFilename)
	if err != nil {
		return err
	}

	sia.EmojiLexiconMap = sia.makeEmojiLexiconMap(string(emojiLexicon))

	//set special case idioms for analyzer
	sia.SpecialCaseIdioms = SpecialCaseIdioms

	return nil
}

//Convert lexicon file to map
func (sia *SentimentIntensityAnalyzer) makeLexiconMap(lexicon string) map[string]float64 {
	lexiconDict := make(map[string]float64)

	for _, line := range strings.Split(lexicon, "\n") {
		line = strings.TrimSpace(line)
		values := strings.Split(line, "\t")

		word := values[0]
		measure, err := strconv.ParseFloat(values[1], 64)
		if err != nil {
			log.Fatal(err)
		}

		lexiconDict[word] = measure
	}

	return lexiconDict
}

// Convert emoji lexicon file to map
func (sia *SentimentIntensityAnalyzer) makeEmojiLexiconMap(emojiLexicon string) map[string]string {
	emojiLexiconDict := make(map[string]string)

	for _, line := range strings.Split(emojiLexicon, "\n") {
		line = strings.TrimSpace(line)
		values := strings.Split(line, "\t")

		word := values[0]
		description := values[1]

		emojiLexiconDict[word] = description
	}

	return emojiLexiconDict
}

//additional function for emoji check
//for case if they are not separated by whitespace
func checkEmojisInText(text string) string {
	// find all emojis in text
	regexAllEmoji := `[\x{1F300}-\x{1F6FF}|[\x{2600}-\x{26FF}]`
	re := regexp.MustCompile(regexAllEmoji)

	emojis := re.FindAllString(text, -1)
	emojisText := strings.Join(emojis, " ")

	//concatenate emojis separated by whitespace with cleaned text
	cleanText := re.ReplaceAllString(text, "")
	text = cleanText + " " + emojisText

	return text
}

// find percent difference occurences (+2%,-2% etc.)
// and replace it with placeholder from lexicon
func checkPercentsInText(text string) string {
	rePos := regexp.MustCompile(`(\(|\s)*(\+(\d+|\d+(\.|\,)\d+)(\%|\s\%))(\)|\s)*`)
	reNeg := regexp.MustCompile(`(\(|\s)*(\-(\d+|\d+(\.|\,)\d+)(\%|\s\%))(\)|\s)*`)

	text = rePos.ReplaceAllString(text, " xpositivepercentx ")
	text = reNeg.ReplaceAllString(text, " xnegativepercentx ")

	return text
}

// PolarityScores Return a float for sentiment strength based on the input text.
// Positive values are positive valence, negative value are negative valence.
func (sia *SentimentIntensityAnalyzer) PolarityScores(text string) map[string]float64 {
	var textNoEmojiList []string

	text = checkPercentsInText(text)
	text = checkEmojisInText(text)
	textTokensList := strings.Fields(text)
	for _, token := range textTokensList {
		if description, ok := sia.EmojiLexiconMap[token]; ok {
			textNoEmojiList = append(textNoEmojiList, description)
		} else {
			textNoEmojiList = append(textNoEmojiList, token)
		}
	}
	text = strings.Join(textNoEmojiList, " ")

	sentiText := SentiText{}
	sentiText.Init(text)

	var sentiments []float64

	wordsAndEmoticons := sentiText.WordsAndEmoticons
	for i, item := range sentiText.WordsAndEmoticons {
		valence := 0.0

		// check for vader_lexicon words that may be used as modifiers or negations
		if _, ok := BoosterMap[strings.ToLower(item)]; ok {
			sentiments = append(sentiments, valence)
			continue
		}

		if i < len(wordsAndEmoticons)-1 && strings.ToLower(item) == "kind" && strings.ToLower(wordsAndEmoticons[i+1]) == "of" {
			sentiments = append(sentiments, valence)
			continue
		}

		sentiments = sia.sentimentValence(valence, &sentiText, item, i, sentiments)
	}

	sentiments = sia.butCheck(wordsAndEmoticons, sentiments)
	valenceDict := sia.scoreValence(sentiments, text)

	return valenceDict
}

func (sia *SentimentIntensityAnalyzer) sentimentValence(valence float64, sentiText *SentiText, item string, i int, sentiments []float64) []float64 {
	isCapDiff := sentiText.IsCapDiff
	wordsAndEmoticons := sentiText.WordsAndEmoticons
	itemLowercase := strings.ToLower(item)

	//get the sentiment valence
	if value, ok := sia.LexiconMap[itemLowercase]; ok {
		valence = value

		//check if sentiment laden word is in ALL CAPS (while others aren't)
		if item == strings.ToUpper(item) && isCapDiff {
			if valence > 0 {
				valence += cIncr
			} else {
				valence -= cIncr
			}
		}

		for start := 0; start <= 2; start++ {
			//// dampen the scalar modifier of preceding words and emoticons
			//// (excluding the ones that immediately preceed the item) based
			//// on their distance from the current item.

			if i <= start {
				continue
			}

			if _, ok := sia.LexiconMap[strings.ToLower(wordsAndEmoticons[i-(start+1)])]; !ok {
				s := scalarIncDec(wordsAndEmoticons[i-(start+1)], valence, isCapDiff)

				if start == 1 && s != 0 {
					s *= 0.95
				}

				if start == 2 && s != 0 {
					s *= 0.9
				}

				valence += s
				valence = sia.negationCheck(valence, wordsAndEmoticons, start, i)

				if start == 2 {
					valence = sia.specialIdiomsCheck(valence, wordsAndEmoticons, i)
				}
			}
		}
		valence = sia.leastCheck(valence, wordsAndEmoticons, i)
	}

	sentiments = append(sentiments, valence)
	return sentiments
}

func (sia *SentimentIntensityAnalyzer) leastCheck(valence float64, wordsAndEmoticons []string, i int) float64 {
	// check for negation case using "least"

	if i > 1 {
		if _, ok := sia.LexiconMap[strings.ToLower(wordsAndEmoticons[i-1])]; !ok && strings.ToLower(wordsAndEmoticons[i-1]) == "least" {
			if strings.ToLower(wordsAndEmoticons[i-2]) != "at" && strings.ToLower(wordsAndEmoticons[i-2]) != "very" {
				valence = valence * nScalar
			}
		}
	} else if i > 0 {
		if _, ok := sia.LexiconMap[strings.ToLower(wordsAndEmoticons[i-1])]; !ok && strings.ToLower(wordsAndEmoticons[i-1]) == "least" {
			valence = valence * nScalar
		}
	}

	return valence
}

func (sia *SentimentIntensityAnalyzer) butCheck(wordsAndEmoticons []string, sentiments []float64) []float64 {
	// check for modification in sentiment due to contrastive conjunction 'but'
	var wordsAndEmoticonsLower []string

	for _, w := range wordsAndEmoticons {
		wordsAndEmoticonsLower = append(wordsAndEmoticonsLower, strings.ToLower(w))
	}

	for bi, wl := range wordsAndEmoticonsLower {
		if wl == "but" {
			for si, sentiment := range sentiments {
				if si < bi {
					sentiments[si] = sentiment * 0.5
				} else if si > bi {
					sentiments[si] = sentiment * 1.5
				}
			}
		}
	}

	return sentiments
}

func (sia *SentimentIntensityAnalyzer) specialIdiomsCheck(valence float64, wordsAndEmoticons []string, i int) float64 {
	var wordsAndEmoticonsLower []string

	for _, w := range wordsAndEmoticons {
		wordsAndEmoticonsLower = append(wordsAndEmoticonsLower, strings.ToLower(w))
	}

	onezero := fmt.Sprintf("%s %s", wordsAndEmoticonsLower[i-1], wordsAndEmoticonsLower[i])

	twoonezero := fmt.Sprintf("%s %s %s", wordsAndEmoticonsLower[i-2], wordsAndEmoticonsLower[i-1], wordsAndEmoticonsLower[i])

	twoone := fmt.Sprintf("%s %s", wordsAndEmoticonsLower[i-2], wordsAndEmoticonsLower[i-1])

	threetwoone := fmt.Sprintf("%s %s %s", wordsAndEmoticonsLower[i-3], wordsAndEmoticonsLower[i-2], wordsAndEmoticonsLower[i-1])

	threetwo := fmt.Sprintf("%s %s", wordsAndEmoticonsLower[i-3], wordsAndEmoticonsLower[i-2])

	sequences := []string{onezero, twoonezero, twoone, threetwoone, threetwo}

	for _, seq := range sequences {
		if value, ok := sia.SpecialCaseIdioms[seq]; ok {
			valence = value
			break
		}
	}

	if len(wordsAndEmoticonsLower)-1 > i {
		zeroone := fmt.Sprintf("%s %s", wordsAndEmoticonsLower[i], wordsAndEmoticonsLower[i+1])
		if value, ok := sia.SpecialCaseIdioms[zeroone]; ok {
			valence = value
		}
	}

	if len(wordsAndEmoticonsLower)-1 > i+1 {
		zeroonetwo := fmt.Sprintf("%s %s %s", wordsAndEmoticonsLower[i], wordsAndEmoticonsLower[i+1], wordsAndEmoticonsLower[i+2])
		if value, ok := sia.SpecialCaseIdioms[zeroonetwo]; ok {
			valence = value
		}
	}

	// check for booster/dampener bi-grams such as 'sort of' or 'kind of'
	ngrams := []string{threetwoone, threetwo, twoone}
	for _, ngram := range ngrams {
		if value, ok := BoosterMap[ngram]; ok {
			valence = valence + value
		}
	}

	return valence
}

// Future Work
// check for sentiment laden idioms that don't contain a lexicon word
func (sia *SentimentIntensityAnalyzer) sentimentLadenIdiomsCheck(valence float64, text string) float64 {
	// TODO in future

	return 0.0
}

//check for negations
func (sia *SentimentIntensityAnalyzer) negationCheck(valence float64, wordsAndEmoticons []string, start int, i int) float64 {
	var wordsAndEmoticonsLower []string

	for _, w := range wordsAndEmoticons {
		wordsAndEmoticonsLower = append(wordsAndEmoticonsLower, strings.ToLower(w))
	}

	if start == 0 {
		if negated([]string{wordsAndEmoticonsLower[i-(start+1)]}) { // 1 word preceding lexicon word (w/o stopwords)
			valence = valence * nScalar
		}
	}

	if start == 1 {
		if wordsAndEmoticonsLower[i-2] == "never" && (wordsAndEmoticonsLower[i-1] == "so" || wordsAndEmoticonsLower[i-1] == "this") {
			valence = valence * 1.25
		} else if wordsAndEmoticonsLower[i-2] == "without" && wordsAndEmoticonsLower[i-1] == "doubt" {
			valence = valence
		} else if negated([]string{wordsAndEmoticonsLower[i-(start+1)]}) { // 2 words preceding the lexicon word position
			valence = valence * nScalar
		}
	}

	if start == 2 {
		if wordsAndEmoticonsLower[i-3] == "never" && (wordsAndEmoticonsLower[i-2] == "so" || wordsAndEmoticonsLower[i-2] == "this") || (wordsAndEmoticonsLower[i-1] == "so" || wordsAndEmoticonsLower[i-1] == "this") {
			valence = valence * 1.25
		} else if wordsAndEmoticonsLower[i-3] == "without" && (wordsAndEmoticonsLower[i-2] == "doubt" || wordsAndEmoticonsLower[i-1] == "doubt") {
			valence = valence
		} else if negated([]string{wordsAndEmoticonsLower[i-(start+1)]}) { //3 words preceding the lexicon word position
			valence = valence * nScalar
		}
	}

	return valence
}

// add emphasis from exclamation points and question marks
func (sia *SentimentIntensityAnalyzer) punctuationEmphasis(text string) float64 {
	epAmplifier := sia.amplifyEP(text)
	qmAmplifier := sia.amplifyQM(text)

	punctEmphAmplifier := epAmplifier + qmAmplifier
	return punctEmphAmplifier
}

// check for added emphasis resulting from exclamation points (up to 4 of them)
func (sia *SentimentIntensityAnalyzer) amplifyEP(text string) float64 {
	ep := regexp.MustCompile(`!`)
	matches := ep.FindAllStringIndex(text, -1)

	epCount := len(matches)
	if epCount > 4 {
		epCount = 4
	}

	// (empirically derived mean sentiment intensity rating increase for exclamation points)
	epAmplifier := float64(epCount) * 0.292

	return epAmplifier
}

// check for added emphasis resulting from question marks (2 or 3+)
func (sia *SentimentIntensityAnalyzer) amplifyQM(text string) float64 {
	qm := regexp.MustCompile(`\?`)
	matches := qm.FindAllStringIndex(text, -1)

	qmCount := len(matches)
	qmAmplifier := 0.0
	if qmCount > 1 {
		if qmCount <= 3 {
			// (empirically derived mean sentiment intensity rating increase for question marks)
			qmAmplifier = float64(qmCount) * 0.18
		} else {
			qmAmplifier = 0.96
		}
	}

	return qmAmplifier
}

// want separate positive versus negative sentiment scores
func (sia *SentimentIntensityAnalyzer) siftSentimentScores(sentiments []float64) (float64, float64, int) {
	posSum := 0.0
	negSum := 0.0
	neuCount := 0

	for _, sentiment := range sentiments {
		if sentiment > 0 {
			posSum += (sentiment + 1) //compensates for neutral words that are counted as 1
		} else if sentiment < 0 {
			negSum += (sentiment - 1) //when used with math.fabs(), compensates for neutrals
		} else {
			neuCount++
		}
	}

	return posSum, negSum, neuCount
}

func (sia *SentimentIntensityAnalyzer) scoreValence(sentiments []float64, text string) map[string]float64 {
	var compound float64
	var pos float64
	var neg float64
	var neu float64

	if len(sentiments) > 0 {
		sumS := floats.Sum(sentiments)
		// compute and add emphasis from punctuation in text
		punctEmphAmplifier := sia.punctuationEmphasis(text)
		if sumS > 0 {
			sumS += punctEmphAmplifier
		} else if sumS < 0 {
			sumS -= punctEmphAmplifier
		}

		compound = normalize(sumS)
		// discriminate between positive, negative and neutral sentiment scores
		posSum, negSum, neuCount := sia.siftSentimentScores(sentiments)

		if posSum > math.Abs(negSum) {
			posSum += punctEmphAmplifier
		} else if posSum < math.Abs(negSum) {
			negSum -= punctEmphAmplifier
		}

		total := posSum + math.Abs(negSum) + float64(neuCount)

		pos = math.Abs(posSum / total)
		neg = math.Abs(negSum / total)
		neu = math.Abs(float64(neuCount) / total)
	}

	sentimentDict := map[string]float64{
		"pos":      floats.Round(pos, 3),
		"neg":      floats.Round(neg, 3),
		"neu":      floats.Round(neu, 3),
		"compound": floats.Round(compound, 4),
	}

	return sentimentDict
}
