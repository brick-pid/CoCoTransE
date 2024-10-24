main() {
    try{
    let correct = ArrayList<Bool>(
            getMaxTriples(5) == 1,
            getMaxTriples(6) == 4,
            getMaxTriples(10) == 36,
            getMaxTriples(100) == 53361
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