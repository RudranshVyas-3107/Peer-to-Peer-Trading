

# Peer-to-Peer Trading Platform

## Overview

This project is a **decentralized peer-to-peer (P2P) trading platform** that enables users to trade assets securely and transparently without intermediaries. Using **blockchain technology** and **smart contracts**, this platform ensures trust, transparency, and security in every transaction.

## Features

- Blockchain-backed for secure and immutable transaction history.
- Smart contract automation to enforce trade terms.
- Escrow system to safeguard funds until both parties fulfill their obligations.
- Real-time notifications for transaction updates.
- Easy-to-use interface designed with both buyers and sellers in mind.
- Multi-currency support for digital asset trading.

## Technology Stack

- **Backend**: Node.js, Express
- **Blockchain**: Ethereum, Solidity (for smart contract development)
- **Frontend**: React.js
- **Database**: MongoDB
- **Smart Contracts**: Truffle, Web3.js, MetaMask

## Setup and Installation

### Prerequisites

Before you begin, ensure that you have met the following requirements:

- Node.js and npm installed on your machine.
- MongoDB installed and running locally.
- MetaMask browser extension for Ethereum transactions.
- Infura account for Ethereum node connection.

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/RudranshVyas-3107/Peer-to-Peer-Trading.git
   cd Peer-to-Peer-Trading
   ```

2. Install the necessary dependencies:
   ```bash
   npm install
   ```

3. Configure the environment variables by creating a `.env` file:
   ```
   DB_URI=mongodb://localhost:27017/peer_to_peer_trading
   INFURA_PROJECT_ID=<your-infura-project-id>
   ```

4. Compile and migrate the smart contracts:
   ```bash
   truffle migrate --network <network-name>
   ```

5. Start the application server:
   ```bash
   npm start
   ```

6. Open your browser and visit `http://localhost:3000` to access the platform.

## Usage

- Users can register and log in with MetaMask.
- Browse available offers or create new trading offers.
- The built-in escrow system ensures secure transactions until both parties confirm the trade.
- Real-time updates keep users informed of the trade status.

## Future Improvements

- Integration of non-fungible tokens (NFTs) for trading unique assets.
- Rating and feedback system for traders.
- Expansion to support more blockchain networks like Binance Smart Chain.
- Mobile application for easier access.

## Contributing

Contributions are welcome! Please follow the steps below if you'd like to contribute:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -S -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Contact

If you have any questions or feedback, feel free to contact me via:

- **GitHub**: [RudranshVyas-3107](https://github.com/RudranshVyas-3107)
- **Email**: f20202389@hyderabad.bits-pilani.ac.in
