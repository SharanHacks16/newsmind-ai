import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout.jsx";
import Home from "./pages/Home.jsx";
import About from "./pages/About.jsx";
import Analytics from "./pages/Analytics.jsx";
export default function App() { return <Routes><Route element={<Layout />}><Route path="/" element={<Home />} /><Route path="/about" element={<About />} /><Route path="/analytics" element={<Analytics />} /></Route></Routes>; }
