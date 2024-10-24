main() {
    try{
    let correct = ArrayList<Bool>(
            intToMiniRoman(19) == "xix",
            intToMiniRoman(152) == "clii",
            intToMiniRoman(251) == "ccli",
            intToMiniRoman(426) == "cdxxvi",
            intToMiniRoman(500) == "d",
            intToMiniRoman(1) == "i",
            intToMiniRoman(4) == "iv",
            intToMiniRoman(43) == "xliii",
            intToMiniRoman(90) == "xc",
            intToMiniRoman(94) == "xciv",
            intToMiniRoman(532) == "dxxxii",
            intToMiniRoman(900) == "cm",
            intToMiniRoman(994) == "cmxciv",
            intToMiniRoman(1000) == "m"
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