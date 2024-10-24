main() {
    try{
    let correct = ArrayList<Bool>(
            strlen("") == 0,
            strlen("x") == 1,
            strlen("asdasnakj") == 9
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