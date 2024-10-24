main() {
    try{
    let correct = ArrayList<Bool>(
            (removeVowels("") == ""),
            (removeVowels("abcdefghijklm") == "bcdfghjklm"),
            (removeVowels("fedcba") == "fdcb"),
            (removeVowels("eeeee") == ""),
            (removeVowels("acBAA") == "cB"),
            (removeVowels("EcBOO") == "cB"),
            (removeVowels("ybcd") == "ybcd")
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