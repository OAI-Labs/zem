import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Zem Pipeline Configurator",
  description: "Visual Pipeline Configuration Tool for xfmr-zem",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
