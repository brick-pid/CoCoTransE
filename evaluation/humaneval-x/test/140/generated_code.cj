main() {
    try{
    let correct = ArrayList<Bool>(
            fixSpaces("Example") == "Example",
            fixSpaces("Mudasir Hanif ") == "Mudasir_Hanif_",
            fixSpaces("Yellow Yellow  Dirty  Fellow") == "Yellow_Yellow__Dirty__Fellow",
            fixSpaces("Exa   mple") == "Exa-mple",
            fixSpaces("   Exa 1 2 2 mple") == "-Exa_1_2_2_mple"
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