main() {
    try{
    let correct = ArrayList<Bool>(
            getClosestVowel("yogurt") == "u",
            getClosestVowel("full") == "u",
            getClosestVowel("easy") == "",
            getClosestVowel("eAsy") == "",
            getClosestVowel("ali") == "",
            getClosestVowel("bad") == "a",
            getClosestVowel("most") == "o",
            getClosestVowel("ab") == "",
            getClosestVowel("ba") == "",
            getClosestVowel("quick") == "",
            getClosestVowel("anime") == "i",
            getClosestVowel("Asia") == "",
            getClosestVowel("Above") == "o"
    );
if (correct.contains(false)) {
                print(correct)
        }
    }
    catch(e: Exception)
    {
        print("false")
    }
}