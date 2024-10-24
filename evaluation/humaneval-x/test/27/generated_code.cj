main() {
    try{
    let correct = ArrayList<Bool>(
            (flipCase("") == ""),
            (flipCase("Hello!") == "hELLO!"),
            (flipCase("These violent delights have violent ends") == "tHESE VIOLENT DELIGHTS HAVE VIOLENT ENDS")
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