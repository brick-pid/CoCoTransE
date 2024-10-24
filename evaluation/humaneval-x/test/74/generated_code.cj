main() {
    try{
    let correct = ArrayList<Bool>(
            totalMatch(ArrayList<String>(), ArrayList<String>()) == (ArrayList<String>()),
            totalMatch(ArrayList<String>("hi", "admin"), ArrayList<String>("hi", "hi")) == (ArrayList<String>("hi", "hi")),
            totalMatch(ArrayList<String>("hi", "admin"), ArrayList<String>("hi", "hi", "admin", "project")) == (ArrayList<String>("hi", "admin")),
            totalMatch(ArrayList<String>("4"), ArrayList<String>("1", "2", "3", "4", "5")) == (ArrayList<String>("4")),
            totalMatch(ArrayList<String>("hi", "admin"), ArrayList<String>("hI", "Hi")) == (ArrayList<String>("hI", "Hi")),
            totalMatch(ArrayList<String>("hi", "admin"), ArrayList<String>("hI", "hi", "hi")) == (ArrayList<String>("hI", "hi", "hi")),
            totalMatch(ArrayList<String>("hi", "admin"), ArrayList<String>("hI", "hi", "hii")) == (ArrayList<String>("hi", "admin")),
            totalMatch(ArrayList<String>(), ArrayList<String>("this")) == (ArrayList<String>()),
            totalMatch(ArrayList<String>("this"), ArrayList<String>()) == (ArrayList<String>())
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