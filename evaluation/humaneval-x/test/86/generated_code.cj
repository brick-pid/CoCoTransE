main() {
    try{
    let correct = ArrayList<Bool>(
            antiShuffle("Hi") == "Hi",
            antiShuffle("hello") == "ehllo",
            antiShuffle("number") == "bemnru",
            antiShuffle("abcd") == "abcd",
            antiShuffle("Hello World!!!") == "Hello !!!Wdlor",
            antiShuffle("") == "",
            antiShuffle("Hi. My name is Mister Robot. How are you?") == ".Hi My aemn is Meirst .Rboot How aer ?ouy"
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