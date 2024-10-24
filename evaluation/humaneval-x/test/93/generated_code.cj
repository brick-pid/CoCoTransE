main() {
    try{
    let correct = ArrayList<Bool>(
            encode("TEST") == "tgst",
            encode("Mudasir") == "mWDCSKR",
            encode("YES") == "ygs",
            encode("This is a message") == "tHKS KS C MGSSCGG",
            encode("I DoNt KnOw WhAt tO WrItE") == "k dQnT kNqW wHcT Tq wRkTg"
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