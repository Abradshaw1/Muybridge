//
//  SettingsView.swift
//  ElifSemesterThesis
//
//  Created by Elif Basokur on 16.07.2024.
//

import Foundation
import SwiftUI

struct SettingsView: View {
    @Binding var useFrontCamera: Bool

    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Camera")) {
                    Toggle(isOn: $useFrontCamera) {
                        Text("Use Front Camera")
                    }
                }
            }
            .navigationBarTitle("Settings", displayMode: .inline)
        }
    }
}

struct SettingsView_Previews: PreviewProvider {
    @State static var useFrontCamera = false

    static var previews: some View {
        SettingsView(useFrontCamera: $useFrontCamera)
    }
}

