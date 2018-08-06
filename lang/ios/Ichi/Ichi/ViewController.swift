//
//  ViewController.swift
//  Ichi
//
//  Created by V on 7/29/18.
//  Copyright © 2018 V. All rights reserved.
//

import UIKit

class ViewController: UIViewController {
    @IBOutlet weak var labelname: UITextField!
    @IBOutlet weak var buttonname: UIButton!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        labelname.text = "12";
        buttonname.setTitle("444", for: .normal);
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    @IBAction func buttonPressed(sender: AnyObject) {
           labelname.text = "1234";
    }


}

