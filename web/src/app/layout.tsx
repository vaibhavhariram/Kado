import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Kado — Video Bug Reporter",
  description:
    "Upload a narrated screen recording and Kado detects where something breaks, generating timestamped failure reports with expected vs actual behavior.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
