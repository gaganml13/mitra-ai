import React from "react";
import { createRoot } from "react-dom/client";
import MitraDashboard from "../MitraDashboard.jsx";

const container = document.getElementById("root");
const root = createRoot(container);
root.render(
  <React.StrictMode>
    <MitraDashboard />
  </React.StrictMode>
);
